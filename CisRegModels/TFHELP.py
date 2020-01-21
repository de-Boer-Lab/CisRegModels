from threading import Thread;
from CisRegModels import MYUTILS
import numpy as np;
import tensorflow as tf;
import sys
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def cycle_learning_rate(learning_rate_low, learning_rate_high, global_step, period, name=None):
	if global_step is None:
		raise ValueError("global_step is required for cycle_learning_rate.")
	with ops.name_scope(name, "CycleLR",
											[learning_rate_high, learning_rate_low, global_step,
											 period]) as name:
		learning_rate_high = ops.convert_to_tensor(learning_rate_high, name="learning_rate_high")
		learning_rate_low = ops.convert_to_tensor(learning_rate_low, name="learning_rate_low")
		
		dtype = learning_rate_high.dtype
		global_step = math_ops.cast(global_step, dtype)
		period = math_ops.cast(period, dtype)
		step_size = tf.divide(tf.subtract(learning_rate_high,learning_rate_low),period) # increment by this until period is reached, then decrement
		cycle_stage = tf.abs(tf.subtract(tf.floormod(global_step,tf.multiply(math_ops.cast(2.0,dtype),period)),period))
		return tf.add(learning_rate_low, tf.multiply(tf.subtract(period,cycle_stage),step_size), name=name)

#abstract object for polymorphism 
class BasicBatchGetter:
	def getNextBatch(self):
		self.curThread.join();
		curX = self.nextBatchX;
		curY = self.nextBatchY;
		curR = self.numRuns;
		self.curThread = Thread(target = self.prepareNextBatch);
		self.curThread.start()
		return(curX, curY, curR);


#asynchronous batch getting for one hot encoding of sequences
class BatchGetterOneHot(BasicBatchGetter):
	def __init__(self, inFP, batchSize, numRuns, seqLen):
		self.inFP = inFP;
		self.batchSize = batchSize;
		self.numRuns= numRuns;
		self.seqLen= seqLen;
		self.curFH = MYUTILS.smartGZOpen(self.inFP,'r')
		self.curThread = Thread(target = self.prepareNextBatch);
		self.curThread.start()
	def prepareNextBatch(self):
		self.nextBatchX = np.zeros((self.batchSize,4,self.seqLen,1))
		self.nextBatchY = np.zeros((self.batchSize))
		b=0
		while b < self.batchSize:
			line = self.curFH.readline()
			if line =="":
				if self.numRuns==1:
					self.nextBatchX = self.nextBatchX[0:b,:,:,:]
					self.nextBatchY = self.nextBatchY[0:b]
					self.numRuns-=1;
					return;
				self.curFH.close();
				self.curFH = MYUTILS.smartGZOpen(self.inFP,'r')
				self.numRuns-=1;
				line = self.curFH.readline()
			if line is None or line[0]=="#": continue
			curData = np.fromstring(line, dtype=float, sep="\t")
			self.nextBatchY[b]=curData[0];
			self.nextBatchX[b,:,:,0] = curData[1:].reshape((4,self.seqLen))
			b+=1


class BatchGetter(BasicBatchGetter):
	def __init__(self, inFP, batchSize, numRuns,numTFs, numKds):
		self.inFP = inFP;
		self.batchSize = batchSize;
		self.numRuns= numRuns;
		self.numTFs= numTFs;
		self.numKds= numKds;
		self.curFH = MYUTILS.smartGZOpen(self.inFP,'r')
		self.curThread = Thread(target = self.prepareNextBatch);
		self.curThread.start()
	def prepareNextBatch(self):
		self.nextBatchX = np.zeros((self.batchSize,self.numKds,self.numTFs)) +np.log(99999.0);
		self.nextBatchY = np.zeros((self.batchSize))
		b=0
		while b < self.batchSize:
			line = self.curFH.readline()
			if line =="":
				if self.numRuns==1:
					self.nextBatchX = self.nextBatchX[0:b,:,:]
					self.nextBatchY = self.nextBatchY[0:b]
					self.numRuns-=1;
					return;
				self.curFH.close();
				self.curFH = MYUTILS.smartGZOpen(self.inFP,'r')
				self.numRuns-=1;
				line = self.curFH.readline()
			if line is None or line[0]=="#": continue
			curData = line.split("\t");
			self.nextBatchY[b]=float(curData[0]);
			for t in range(1,len(curData)):
				curKds = [np.log(float(x)) for x in curData[t].split(";")]
				self.nextBatchX[b,0:min(self.numKds,len(curKds)),t-1] = curKds[0:min(self.numKds,len(curKds))];
			b+=1

class BatchGetterFixedNumKds(BatchGetter):
	def prepareNextBatch(self):
		self.nextBatchX = np.zeros((self.batchSize,self.numKds,self.numTFs)) +np.log(99999.0);
		self.nextBatchY = np.zeros((self.batchSize))
		b=0
		while b < self.batchSize:
			line = self.curFH.readline()
			if line =="":
				if self.numRuns==1:
					self.nextBatchX = self.nextBatchX[0:b,:,:]
					self.nextBatchY = self.nextBatchY[0:b]
					self.numRuns-=1;
					return;
				self.curFH.close();
				self.curFH = MYUTILS.smartGZOpen(self.inFP,'r')
				self.numRuns-=1;
				line = self.curFH.readline()
			if line is None or line[0]=="#": continue
			curData = np.fromstring(line, dtype=float, sep="\t")
			self.nextBatchY[b]=curData[0];
			self.nextBatchX[b,:,:] = np.transpose(curData[1:len(curData)].reshape((self.numTFs,self.numKds)))
			b+=1


class BatchGetterSeq2Vec(BasicBatchGetter):
	def __init__(self, inFP, batchSize, numRuns,seqLen, kmer2index, wordLen):
		self.inFP = inFP;
		self.batchSize = batchSize;
		self.numRuns= numRuns;
		self.seqLen= seqLen;
		self.wordLen= wordLen;
		self.kmer2index= kmer2index;
		self.curFH = MYUTILS.smartGZOpen(self.inFP,'r')
		self.curThread = Thread(target = self.prepareNextBatch);
		self.curThread.start()
		
	def prepareNextBatch(self):
		self.nextBatchX = np.zeros((self.batchSize,self.seqLen-self.wordLen+1)).astype("int32");
		self.nextBatchY = np.zeros((self.batchSize))
		b=0
		while b < self.batchSize:
			line = self.curFH.readline()
			if line =="":
				if self.numRuns==1:
					self.nextBatchX = self.nextBatchX[0:b,:,:]
					self.nextBatchY = self.nextBatchY[0:b]
					self.numRuns-=1;
					return;
				self.curFH.close();
				self.curFH = MYUTILS.smartGZOpen(self.inFP,'r')
				self.numRuns-=1;
				line = self.curFH.readline()
			if line is None or line[0]=="#": continue
			curData = line.rstrip().split("\t");
			self.nextBatchY[b]=float(curData[0]);
			curSeq = curData[1];
			if len(curSeq) < self.seqLen:
				curSeq = "N"*(self.seqLen - len(curSeq)) + curSeq;  ### prepend Ns if the sequence is too short
			curSeq = curSeq[(len(curSeq)-self.seqLen):len(curSeq)]# trim distal bases if too long
			for si in range(0,self.seqLen-self.wordLen+1):
				self.nextBatchX[b,si] = self.kmer2index[curSeq[si:(si+self.wordLen)]] #fill X with the indeces of the various k-mers
			b+=1


