from CisRegModels import MYUTILS
from CisRegModels import PWM;
from CisRegModels import TFHELP;
import tensorflow as tf
import numpy as np;
from datetime import datetime
import os;
import sys

BASES = ['A','C','G','T']

class CRM:
	#global self.args, self.potentiation, self.constantPot, self.positionalActivityBias, self.sess, self.train_step, self.myLoss, self.mseTF, self.mergedSummaries, self.global_step, self.ohcX, self.realELY, self.summaryWriter,self.epBoundTensor, self.epBoundTensorRC, self.seqPotentialTensor, self.motifsTensor2, self.logConcs, self.activities, self.constant, self.saver, self.saveParams, self.positionalActivityBiasRC, self.activityDiffs, BASES
	
	def __init__(self, args2):
		self.args=args2;
	
	def testModel(self):
		if (self.args.logFP is not None):
			logFile=MYUTILS.smartGZOpen(self.args.logFP,'w');
			sys.stderr=logFile;
		
		if (self.args.outFP is not None):
			if self.args.verbose>0: sys.stderr.write("Outputting to file "+self.args.outFP+"*\n");
			outFile=MYUTILS.smartGZOpen(self.args.outFP,'w');
		else:
			outFile = sys.stdout;
		if self.args.loadModel is None and self.args.restore is None:
			raise Exception("Must specify either -res or -M")
		if self.args.loadModel is not None:
			sys.stderr.write("Loading saved model: %s\n"%(self.args.loadModel));
			self.sess = tf.Session()
			tf.saved_model.loader.load(self.sess, ['main'], self.args.loadModel)
			varNames = [v.name for v in tf.global_variables()];
			predELY = tf.get_default_graph().get_tensor_by_name("predELY:0")
			self.ohcX = tf.get_default_graph().get_tensor_by_name("ohcX:0")
			if self.args.outputBinding>0:
				self.epBoundTensor = tf.get_default_graph().get_tensor_by_name("epBoundTensor:0")
				if "epBoundTensorRC:0" in varNames:
					self.args.trainStrandedActivities=1
					self.epBoundTensorRC = tf.get_default_graph().get_tensor_by_name("epBoundTensorRC:0")
			if "seqPotentialTensor:0" in varNames:
				self.args.potentiation=1
				self.seqPotentialTensor = tf.get_default_graph().get_tensor_by_name("seqPotentialTensor:0")
		else:
			self.makeGraph();
		outFile.write("actual\tpredicted");
		if self.args.potentiation>0:
			outFile.write("\tpred_openness");
		
		if self.args.outputBinding > 0:
			if self.args.trainPositionalActivities > 0:
				raise Exception("Cannot output binding values while using positional self.activities");
			outFile.write("\t" + "\t".join(["Binding_%i"%x for x in range(0,self.args.numMotifs)]))
			if self.args.trainStrandedActivities>0:
				outFile.write("\t" + "\t".join(["RCBinding_%i"%x for x in range(0,self.args.numMotifs)]))
		
		outFile.write("\n");
		
		b = 0
		batchX = np.zeros((self.args.batch,4,self.args.seqLen,1))
		batchY = np.zeros((self.args.batch))
		inFile=MYUTILS.smartGZOpen(self.args.inFP,'r');
		for line in inFile:
			if line is None or line == "" or line[0]=="#": continue
			curData = np.fromstring(line, dtype=float, sep="\t")
			batchY[b]=curData[0];
			batchX[b,:,:,0] = curData[1:].reshape((4,self.args.seqLen))
			b+=1
			if b==self.args.batch:
				#curPredY  = self.sess.run([predELY], feed_dict={self.ohcX: batchX})
				curPredY = predELY.eval(session=self.sess, feed_dict={self.ohcX: batchX})
				if self.args.outputBinding>0:
					bindingAmount = self.epBoundTensor.eval(session=self.sess, feed_dict={self.ohcX: batchX});
					if self.args.trainStrandedActivities>0:
						bindingAmountRC = self.epBoundTensorRC.eval(session=self.sess, feed_dict={self.ohcX: batchX});
				if self.args.potentiation>0:
					curPredOpenness = self.seqPotentialTensor.eval(session=self.sess, feed_dict={self.ohcX: batchX})
				for i in range(0,batchY.shape[0]):
					outFile.write("%g\t%g"%(batchY[i],curPredY[i]));
					if self.args.potentiation>0:
						outFile.write("\t%g"%(curPredOpenness[i]));
					if self.args.outputBinding > 0:
						outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmount[i,]])));
						if self.args.trainStrandedActivities>0:
							outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmountRC[i,]])));
					outFile.write("\n");
				b=0;
		
		inFile.close();

		#test with remaining data, but remove anything past b
		if b > 0:
			batchX = batchX[0:b,:,:,:]
			batchY = batchY[0:b];
			curPredY = predELY.eval(session=self.sess, feed_dict={self.ohcX: batchX})
			if self.args.outputBinding>0:
				bindingAmount = self.epBoundTensor.eval(session=self.sess, feed_dict={self.ohcX: batchX});
				if self.args.trainStrandedActivities>0:
					bindingAmountRC = self.epBoundTensorRC.eval(session=self.sess, feed_dict={self.ohcX: batchX});
			if self.args.potentiation>0:
				curPredOpenness = self.seqPotentialTensor.eval(session=self.sess, feed_dict={self.ohcX: batchX})
			for i in range(0,batchY.shape[0]):
				outFile.write("%g\t%g"%(batchY[i],curPredY[i]));
				if self.args.potentiation>0:
					outFile.write("\t%g"%(curPredOpenness[i]));
				if self.args.outputBinding > 0:
					outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmount[i,]])));
					if self.args.trainStrandedActivities>0:
						outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmountRC[i,]])));
				outFile.write("\n");
		
		self.sess.close()
		sys.stderr.write("Done!\n")
		if (self.args.logFP is not None):
			logFile.close();
		
	
	def makeModel(self):
		if (self.args.logFP is not None):
			logFile=MYUTILS.smartGZOpen(self.args.logFP,'w');
			sys.stderr=logFile;
		
		if (self.args.outFPre is not None):
			if self.args.verbose>0: sys.stderr.write("Outputting to file "+self.args.outFPre+"*\n");
		self.makeGraph()
		
		if self.args.verbose>1: sys.stderr.write("saving session... ");
		save_path = self.saver.save(self.sess, self.args.outFPre+".ckpt")
		self.saveParams(self.sess)
		if self.args.verbose>1: sys.stderr.write("done. saved in %s\n"%save_path);
		
		#remove the file that labels the model as complete
		if os.path.isfile(self.args.outFPre+".done"):
			if self.args.verbose>1: sys.stderr.write("Deleting old \"done\" file %s\n"%(self.args.outFPre+".done"));
			os.remove(self.args.outFPre+".done")
		
		if self.args.tensorboard is not None:
			nextTBF = 1;
		
		globalStep=0;
		if self.args.verbose>1: sys.stderr.write("Running through data %i times\n"%self.args.runs)
		
		batchGetter = TFHELP.BatchGetterOneHot(self.args.inFP, self.args.batch, self.args.runs, self.args.seqLen)
		batchX, batchY, runsLeft = batchGetter.getNextBatch();
		runningMeanMSE = np.zeros(self.args.runningAverageWindow);#init to nans
		runningMeanMSE.fill(np.nan)
		runningMSE=0
		lastTime = datetime.now();
		curTimeNonTF = datetime.now()
		#main loop of training and occasionally saving sessions/summaries
		while runsLeft>0:
			#train with current data
			if self.args.verbose>3: sys.stderr.write("  Time for last non-TF code: %f\n"%(datetime.now()-curTimeNonTF).total_seconds())
			curTimeTF = datetime.now()
			if self.args.trace >0 and globalStep>100: #don't take the first one
				run_metadata = tf.RunMetadata()
				_, curLoss, curMSE, summaries, globalStep = self.sess.run([self.train_step, self.myLoss, self.mseTF, self.mergedSummaries, self.global_step], feed_dict={self.ohcX: batchX, self.realELY: batchY}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
				from tensorflow.python.client import timeline
				trace = timeline.Timeline(step_stats=run_metadata.step_stats)
				trace_file = open('%s.trace.json'%self.args.outFPre, 'w')
				trace_file.write(trace.generate_chrome_trace_format())
				quit();
			elif self.args.tensorboard is not None and ((self.args.tensorboardFrequency is None and nextTBF == (globalStep+1)) or (self.args.tensorboardFrequency is not None and (globalStep+1) % self.args.tensorboardFrequency==0 )):
				sys.stderr.write("  Getting tensorboard summaries\n")
				
				_, curLoss, curMSE, summaries, globalStep = self.sess.run([self.train_step, self.myLoss, self.mseTF, self.mergedSummaries, self.global_step], feed_dict={self.ohcX: batchX, self.realELY: batchY})
			else:
				_, curLoss, curMSE, globalStep = self.sess.run([self.train_step, self.myLoss, self.mseTF, self.global_step], feed_dict={self.ohcX: batchX, self.realELY: batchY})
			if self.args.verbose>3: sys.stderr.write("  Time for last TF code: %f\n"%(datetime.now()-curTimeTF).total_seconds())
			curTimeNonTF = datetime.now()
			runningMeanMSE[globalStep % self.args.runningAverageWindow]=curMSE;
			runningMSE = np.nanmean(runningMeanMSE);
			if self.args.verbose>3 or (self.args.verbose>2 and (globalStep % 100) ==0) or (self.args.verbose>1 and (globalStep % 10000) ==0): 
				curTime = datetime.now()
				deltaTime = (curTime-lastTime).total_seconds()
				lastTime=curTime;
				if self.args.L1 is not None:
					sys.stderr.write("	Batch = %i; examples = %i;\trunning MSE = %.3f;\tlast MSE = %.2f;\tlast time = %.2f\tloss = %.2f;\t paramsum = %.2f;\tparamnum = %i\n"%(globalStep, globalStep * self.args.batch,  runningMSE,curMSE, deltaTime,curLoss,paramPenaltyL1Tensor.eval(session=self.sess), paramNumActivityTensor.eval(session=self.sess)));
				else:
					sys.stderr.write("	Batch = %i; examples = %i;\trunning MSE = %.3f;\tlast MSE = %.2f\tlast time = %.2f\n"%(globalStep, globalStep * self.args.batch, runningMSE,curMSE, deltaTime));
			#save state, if desired
			if self.args.tensorboard is not None and ((self.args.tensorboardFrequency is None and nextTBF == globalStep) or (self.args.tensorboardFrequency is not None and globalStep % self.args.tensorboardFrequency==0 )):
				nextTBF = globalStep*2;
				if self.args.verbose>1: sys.stderr.write("	saving tensorboard summary %s... "%globalStep);
				self.summaryWriter.add_summary(summaries, globalStep)
				if self.args.verbose>1: sys.stderr.write("done\n");
			if globalStep % self.args.saveEvery==0:
				if np.isnan(curMSE):
					raise Exception("ERROR: reached nan MSE - quitting without saving.");
				if self.args.verbose>1: sys.stderr.write("	saving session... ");
				save_path = self.saver.save(self.sess, self.args.outFPre+".ckpt")
				self.saveParams(self.sess)
				if self.args.verbose>1: sys.stderr.write("done. saved in %s\n"%save_path);
			curTimeB = datetime.now()
			batchX, batchY, runsLeft = batchGetter.getNextBatch();
			if self.args.verbose>3: sys.stderr.write("  Time To Get Batch: %f\n"%(datetime.now()-curTimeB).total_seconds())
			
		#train on the final batch
		if self.args.tensorboard is not None:
			_, curLoss, curMSE, summaries = self.sess.run([self.train_step, self.myLoss, self.mseTF, self.mergedSummaries], feed_dict={self.ohcX: batchX, self.realELY: batchY})
			if self.args.verbose>1: sys.stderr.write("Final saving tensorboard summary... ");
			self.summaryWriter.add_summary(summaries, globalStep)
			if self.args.verbose>1: sys.stderr.write("done\n");
		else:
			self.sess.run(self.train_step, feed_dict={self.ohcX: batchX, self.realELY: batchY})
		
		#final session saving
		if self.args.verbose>1: sys.stderr.write("Final saving session... ");
		save_path = self.saver.save(self.sess, self.args.outFPre+".ckpt")
		self.saveParams(self.sess)
		
		if self.args.verbose>1: sys.stderr.write("saving model... ");
		modelSaveDir = "%s.Model"%self.args.outFPre;
		if self.args.verbose>1: sys.stderr.write("done. saved in %s\n"%save_path);
		if os.path.exists(modelSaveDir):
			import shutil
			if self.args.verbose>1: sys.stderr.write(" deleting old model directory... ");
			shutil.rmtree(modelSaveDir)
		builder = tf.saved_model.builder.SavedModelBuilder(modelSaveDir)
		builder.add_meta_graph_and_variables(self.sess, ["main"])#, signature_def_map=foo_signatures, assets_collection=foo_assets)
		# Add a second MetaGraphDef for inference.
		builder.save()
		if self.args.verbose>1: sys.stderr.write("done. saved in %s\n"%modelSaveDir);
		
		open(self.args.outFPre+".done", 'a').close()
		
		self.sess.close()
		if (self.args.logFP is not None):
			logFile.close();
	
	def makeGraph(self):
		sys.stderr.write("Input arguments: " + " ".join(sys.argv)+"\n")
		
		if self.args.accIsAct>0 and self.args.potentiation==0:
			raise Exception("Cannot include accessibility influence on EL without including self.potentiation layer!")
		
		if not hasattr(self.args, 'VARIABLE'):
			self.args.VARIABLE=None;
		if not hasattr(self.args, 'tensorboard'):
			self.args.tensorboard=None;
		if not hasattr(self.args, 'learningRateED'):
			self.args.learningRateED=1.0;
		if not hasattr(self.args, 'cycleLearningRate'):
			self.args.cycleLearningRate=None;
		if not hasattr(self.args, 'trace'):
			self.args.trace=0;
		if not hasattr(self.args, 'useMomentumOptimizer'):
			self.args.useMomentumOptimizer=0;
			self.args.useNesterov=0;
		if not hasattr(self.args, 'useRMSProp'):
			self.args.useRMSProp=0;
			if not hasattr(self.args, 'rmsEpsilon'):
				self.args.rmsEpsilon=1e-10;
			if not hasattr(self.args, 'momentum'):
				self.args.momentum=0.0;
			if not hasattr(self.args, 'rmsDecay'):
				self.args.rmsDecay=0.9;
		if not hasattr(self.args, 'Abeta1'):
			self.args.Abeta1=0.9;
		if not hasattr(self.args, 'Abeta2'):
			self.args.Abeta2=0.999;
		if not hasattr(self.args, 'Aepsilon'):
			self.args.Aepsilon=1E-8;
		if not hasattr(self.args, 'clearAdamVars'):
			self.args.clearAdamVars=0;
		if not hasattr(self.args, 'L2Pos'):
			self.args.L2Pos=0.001;
		if not hasattr(self.args, 'L2'):
			self.args.L2=0.001;
		if not hasattr(self.args, 'L1'):
			self.args.L1=0.001;
		if not hasattr(self.args, 'learningRate'):
			self.args.learningRate=0.001;
		if not hasattr(self.args, 'meanSDFile'):
			self.args.meanSDFile=None;
		
		### set up tensorflow graph
		#useful: tf.Tensor.get_shape(predELY)
		#sequence layer
		self.ohcX = tf.placeholder(tf.float32, [None,4,self.args.seqLen,1], name="ohcX") # here, [None, 4,...] indicate that the first dimention is of unknown length
		#motif parameters:
		#allow for the user to provide a list of PWMs that we initialize the motifs tensor to
		if self.args.motifsFP is not None:
			if self.args.verbose > 0: sys.stderr.write("Reading in default set of TF Motifs from %s\n"%(self.args.motifsFP));
			#motifsFP is a one-per-line list of PWM paths
			motifsFile=MYUTILS.smartGZOpen(self.args.motifsFP,'r');
			defaultMotifs = np.random.normal(0,1,[4,self.args.motifLen,self.args.numMotifs])
			i=0
			for line in motifsFile:
				if line is None or line == "" or line[0]=="#": continue
				curPWM = PWM.loadPWM(line.rstrip());
				pwmLen = curPWM.len();
				pwmStart = 0;
				pwmEnd = pwmLen; # actually end index+1
				if pwmLen > self.args.motifLen: # can't fit the motif; 
					#trim the terminal bases until pwmEnd ==self.args.motifLen; greedy
					while (pwmEnd - pwmStart) > self.args.motifLen:
						startInfo = 0.0;
						endInfo = 0.0; 
						for b in BASES:
							startInfo += curPWM.mat[b][pwmStart]**2
							endInfo += curPWM.mat[b][pwmEnd-1]**2
						if startInfo > endInfo:
							pwmEnd-=1;
						else:
							pwmStart+=1;
				paddingFront = int((self.args.motifLen - (pwmEnd - pwmStart))/2); #truncated 
				for bi in range(0,len(BASES)):
					b=BASES[bi];
					defaultMotifs[bi,:,i] = 0.0 # set other entries to 0
					defaultMotifs[bi,paddingFront:(paddingFront+pwmEnd-pwmStart),i] = curPWM.mat[b][pwmStart:pwmEnd];
				i+=1;
			if self.args.noTrainMotifs>0:
				motifsTensor = tf.constant(defaultMotifs.astype(np.float32),name="motifs") 
			else:
				motifsTensor = tf.Variable(tf.convert_to_tensor(defaultMotifs.astype(np.float32)),name="motifs") 
		else:
			if self.args.noTrainMotifs>0:
				motifsTensor = tf.constant(np.random.standard_normal([4,self.args.motifLen,self.args.numMotifs]).astype(np.float32),name="motifs") 
			else:
				motifsTensor = tf.Variable(tf.random_normal([4,self.args.motifLen,self.args.numMotifs]),name="motifs") 
		
		motifsRCTensor = tf.image.flip_left_right(tf.image.flip_up_down(motifsTensor)) #flips along first two dims; this works because order of bases is ACGT=rc(TGCA)
		
		self.motifsTensor2 = tf.reshape(motifsTensor, [4,self.args.motifLen,1,self.args.numMotifs]);#patch height, patch width, input channels, output channels
		motifsRCTensor2 = tf.reshape(motifsRCTensor, [4,self.args.motifLen,1,self.args.numMotifs]);
		#output: [4, motifLen, 1, numMotifs]
		
		############## INPUT ANY PROVIDED INITIAL PARAMS
		## ACTIVITIES
		initActiv = np.random.standard_normal(self.args.numMotifs).astype(np.float32);
		#input default concentrations if applicable
		if self.args.initActiv is not None:
			if self.args.verbose>0: sys.stderr.write("Initializing self.activities to %s\n"%self.args.initActiv)
			try: 
				vals = np.loadtxt(self.args.initActiv, dtype = np.str);
				initActiv[0:vals.shape[0]] = vals[:,1].astype("float32");
			except IOError:
				initActiv = np.zeros([self.args.numMotifs]).astype("float32")+float(self.args.initActiv)
		
		
		if self.args.noTrainActivities==0:
			if self.args.verbose>0: sys.stderr.write("Training self.activities\n")
			self.activities = tf.Variable(tf.convert_to_tensor(initActiv.reshape([self.args.numMotifs])),name="activities");
		else:
			if self.args.verbose>0: sys.stderr.write("Not training self.activities\n")
			self.activities = tf.constant(initActiv.reshape([self.args.numMotifs]),name="activities");
		
		if self.args.trainStrandedActivities>0:
			if self.args.verbose>0: sys.stderr.write("Training stranded self.activities\n")
			self.activityDiffs = tf.Variable(tf.zeros(self.args.numMotifs), name="activityDiffs")
			activitiesRC = tf.add(self.activities, self.activityDiffs);
		
		if self.args.trainPositionalActivities>0:
			if self.args.verbose>0: sys.stderr.write("Training positionally biased self.activities\n")
			self.positionalActivityBias = tf.Variable(tf.ones([self.args.seqLen,self.args.numMotifs]), name="positionalActivities") #[seqLen, numMotifs]
			positionalActivity = tf.multiply(self.positionalActivityBias, self.activities) #[seqLen, numMotifs]
			if self.args.trainStrandedActivities:
				self.positionalActivityBiasRC = tf.Variable(tf.ones([self.args.seqLen,self.args.numMotifs]), name="positionalActivitiesRC") #[seqLen, numMotifs]
				positionalActivityRC = tf.multiply(self.positionalActivityBiasRC, activitiesRC) #[seqLen, numMotifs]
		
		## CONCENTRATIONS
		initConcs = np.zeros(self.args.numMotifs).astype(np.float32);
		#input default concentrations if applicable
		if self.args.initConcs is not None:
			if self.args.verbose>0: sys.stderr.write("Initializing concentrations to %s\n"%self.args.initConcs)
			vals = np.loadtxt(self.args.initConcs, dtype = np.str);
			initConcs[0:vals.shape[0]] = vals[:,1].astype("float32");
		
		
		if self.args.noTrainConcs==0:
			if self.args.verbose>0: sys.stderr.write("Training concentrations\n")
			if not self.args.VARIABLE:
				self.logConcs = tf.Variable(tf.convert_to_tensor(initConcs.reshape((1,1,1,self.args.numMotifs))), name="concentrations"); 
			else:
				self.logConcs = tf.Variable(tf.convert_to_tensor(initConcs.reshape((1,1,1,self.args.numMotifs))), name="Variable"); 
		else:
			if self.args.verbose>0: sys.stderr.write("Not training concentrations\n")
			self.logConcs = tf.constant(initConcs.reshape((1,1,1,self.args.numMotifs)), name="concentrations"); ###TODO name="concentrations"
		
		## binding limits
		if self.args.bindingLimits:
			initBL = np.ones(self.args.numMotifs).astype(np.float32);
			#input default concentrations if applicable
			if self.args.initBindLim is not None:
				if self.args.verbose>0: sys.stderr.write("Initializing binding limits to %s\n"%self.args.initBindLim)
				vals = np.loadtxt(self.args.initBindLim, dtype = np.str);
				initBL[0:vals.shape[0]] = np.log(vals[:,1].astype("float32"));  #thus they are input as non-log
		
			if self.args.noTrainBL==0:
				if self.args.verbose>0: sys.stderr.write("Training binding limits\n")
				logBindingLimits = tf.Variable(tf.convert_to_tensor(initBL.reshape((self.args.numMotifs))), name="logBindingLimits");  #[numMotifs]
			else:
				if self.args.verbose>0: sys.stderr.write("Not training binding limits\n")
				logBindingLimits = tf.constant(initBL.reshape((self.args.numMotifs)), name="logBindingLimits");  #[numMotifs]
			bindingLimits=tf.exp(logBindingLimits, name="bindingLimits");
		
		#motif layer: conv layer 1
			#nodes: motifs * orientations * positions
			#params: motifs * motifLens * 4
		#strides all =1 - this is the step size
		# zero padding =SAME makes output dims =input; valid does not pad with 0s
		#motifScanTensor = tf.nn.conv2d(self.ohcX, self.motifsTensor2, strides = [1,1,1,1], padding='VALID', name="motifScan") #VALID so that the output dimensions are 1 * seqLen-motifLen+1, ...
		#motifScanRCTensor= tf.nn.conv2d(self.ohcX,motifsRCTensor2, strides = [1,1,1,1], padding='VALID', name="motifScanRC") #VALID so that the output dimensions are 1 * seqLen-motifLen+1, ...
		#### ##outputs [None,1,seqLen-motifLen+1,numMotifs]
		#these are log(Kds)
		motifScanTensor = tf.nn.conv2d(self.ohcX, self.motifsTensor2, strides = [1,4,1,1], padding='SAME', name="motifScan") 
		motifScanRCTensor= tf.nn.conv2d(self.ohcX,motifsRCTensor2, strides = [1,4,1,1], padding='SAME', name="motifScanRC") 
		##outputs [None,1,seqLen,numMotifs]
		
		logKdConcRatioTensor = tf.subtract(self.logConcs,motifScanTensor) # [None, 1, seqLen,numMotifs] 
		logKdConcRatioRCTensor = tf.subtract(self.logConcs,motifScanRCTensor) # [None, 1, seqLen,numMotifs] 
		# BUG: the below two lines should technically use 2^x instead of tf.exp (which uses e) since the PKdMs are log base 2
		# but this does not affect the ability of the model to learn because the difference in base can be absorbed by the parameters of the PKdM and concentration
		pNotBoundTensor1 = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioTensor))); # size: [None,1,seqLen,numMotifs]
		pNotBoundRCTensor1 = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioRCTensor))); # size: [None,1,seqLen,numMotifs]
		pNotBoundTensor = pNotBoundTensor1;
		pNotBoundRCTensor = pNotBoundRCTensor1
		
		if self.args.useEBound>0: 
			if self.args.verbose>0: sys.stderr.write("Using E-bound\n")
			self.epBoundTensor = tf.add(tf.reduce_sum(tf.subtract(1.0,pNotBoundRCTensor), reduction_indices=[1,2]),tf.reduce_sum(tf.subtract(1.0,pNotBoundTensor), reduction_indices=[1,2])) # size: [None, numMotifs] #expected amount of binding
		else:
			if self.args.verbose>0: sys.stderr.write("Using P-bound\n")
			self.epBoundTensor = tf.subtract(1.0,tf.multiply(tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2]),tf.reduce_prod(pNotBoundTensor, reduction_indices=[1,2]))) # size: [None, numMotifs] # p(bound)
		
		## POTENTIATION
		if self.args.potentiation>0:
			if self.args.verbose>0: sys.stderr.write("Using self.potentiation layer\n")
			initPotent = np.random.standard_normal(self.args.numMotifs).astype(np.float32);
			#input default concentrations if applicable
			if self.args.initPotent is not None:
				if self.args.verbose>0: sys.stderr.write("Initializing potentiations to %s\n"%self.args.initPotent)
				try: 
					vals = np.loadtxt(self.args.initPotent, dtype = np.str);
					initPotent[0:vals.shape[0]] = vals[:,1].astype("float32");
				except IOError:
					#initPotent = tf.fill([self.args.numMotifs],float(self.args.initPotent))
					initPotent = np.zeros([self.args.numMotifs]).astype("float32") + float(self.args.initPotent)
			if self.args.noTrainPotentiations==0:
				if self.args.verbose>0: sys.stderr.write("Training potentiations\n")
				self.potentiation = tf.Variable(tf.convert_to_tensor(initPotent.reshape([self.args.numMotifs])),name="potents");
			else:
				if self.args.verbose>0: sys.stderr.write("Not training potentiations\n")
				self.potentiation = tf.constant(initPotent.reshape([self.args.numMotifs]),name="potents");
			seqPotentialByTFTensor = tf.multiply(self.epBoundTensor, self.potentiation); #size: [None,numMotifs]
			self.constantPot = tf.Variable(tf.zeros(1),name="constantPot")
			self.seqPotentialTensor = tf.sigmoid(tf.add(tf.reduce_sum(seqPotentialByTFTensor,reduction_indices=[1]), self.constantPot), name="seqPotentialTensor") #[None, 1]
		else:
			if self.args.verbose>0: sys.stderr.write("Not using self.potentiation layer\n")
		
		if self.args.trainPositionalActivities>0: # account for positional activity with linear scaling of activity
			if self.args.trainStrandedActivities>0: # account for strand-specific activity biases
				pBoundPerPos = tf.subtract(1.0,pNotBoundTensor) # size: [None,1,seqLen,numMotifs]
				pBoundPerPosRC = tf.subtract(1.0,pNotBoundRCTensor) # size: [None,1,seqLen,numMotifs]
				if self.args.potentiation>0:
					pBoundPerPos = tf.transpose(tf.multiply(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), self.seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
					pBoundPerPosRC = tf.transpose(tf.multiply(tf.transpose(pBoundPerPosRC, perm=(1,2,3,0)), self.seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
				#print(tf.Tensor.get_shape(pBoundPerPos))
				#print(tf.Tensor.get_shape(positionalActivity))
				if self.args.bindingLimits>0:
					expectedActivitySense = tf.reduce_sum(tf.multiply(pBoundPerPos, positionalActivity),reduction_indices=[1,2]) # size: [None,numMotifs]
					expectedActivityRC = tf.reduce_sum(tf.multiply(pBoundPerPosRC, positionalActivityRC),reduction_indices=[1,2]) # size: [None,numMotifs]
					expectedActivityPerTF = tf.add( #min of positive self.activities and max of negative self.activities accounting for binding limits.
						tf.nn.relu(                        tf.minimum(tf.reshape(tf.multiply(bindingLimits,self.activities),(1,self.args.numMotifs)),tf.add(expectedActivitySense, expectedActivityRC))), #positive
						tf.negative(tf.nn.relu(tf.negative(tf.maximum(tf.reshape(tf.multiply(bindingLimits,self.activities),(1,self.args.numMotifs)),tf.add(expectedActivitySense, expectedActivityRC))))) #negative
						) #[None,numMotifs]
					
				else:
					#expectedActivitySense = tf.multiply(tf.reshape(pBoundPerPos, (-1, self.args.seqLen*self.args.numMotifs)), tf.reshape(positionalActivity, (self.args.seqLen*self.args.numMotifs,1))) # size: [None,numMotifs]
					#expectedActivityRC = tf.multiply(tf.reshape(pBoundPerPosRC, (-1, self.args.seqLen*self.args.numMotifs)), tf.reshape(positionalActivityRC, (self.args.seqLen*self.args.numMotifs,1))) # size: [None,1]
					expectedActivitySense = tf.reduce_sum(tf.multiply(pBoundPerPos, positionalActivity), reduction_indices=[1,2]) # size: [None,numMotifs]
					#print(tf.Tensor.get_shape(expectedActivitySense))
					expectedActivityRC = tf.reduce_sum(tf.multiply(pBoundPerPosRC, positionalActivityRC), reduction_indices=[1,2])  # size: [None,numMotifs]
					expectedActivityPerTF = tf.add(expectedActivitySense, expectedActivityRC);
			else:
				if self.args.useEBound>0:
					pBoundPerPos = tf.add(tf.subtract(1.0,pNotBoundTensor), tf.subtract(1.0,pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
				else:
					pBoundPerPos = tf.subtract(1.0,tf.multiply(pNotBoundTensor, pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
				#print(tf.Tensor.get_shape(pBoundPerPos))
				if self.args.potentiation>0:
					pBoundPerPos = tf.transpose(tf.multiply(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), self.seqPotentialTensor), perm = (3,0,1,2)) # size: [None,1,seqLen,numMotifs]
				if self.args.bindingLimits>0:
					pBoundPerPos = tf.minimum(pBoundPerPos, bindingLimits); #[None,1,seqLen,numMotifs]
				#print(tf.Tensor.get_shape(pBoundPerPos))
				#print(tf.Tensor.get_shape(positionalActivity))
				if self.args.bindingLimits>0:
					expectedActivitySense = tf.reduce_sum(tf.multiply(pBoundPerPos, positionalActivity),reduction_indices=[1,2]) # size: [None,numMotifs]
					expectedActivityPerTF = tf.add( #min of positive self.activities and max of negative self.activities accounting for binding limits.
						tf.nn.relu(                        tf.minimum(tf.reshape(tf.multiply(bindingLimits,self.activities),(1,self.args.numMotifs)),expectedActivitySense)), #positive
						tf.negative(tf.nn.relu(tf.negative(tf.maximum(tf.reshape(tf.multiply(bindingLimits,self.activities),(1,self.args.numMotifs)),expectedActivitySense)))) #negative
						) #[None,numMotifs]
				else:
					expectedActivityPerTF = tf.multiply(tf.reshape(pBoundPerPos, (-1, self.args.seqLen*self.args.numMotifs)), tf.reshape(positionalActivity, (self.args.seqLen*self.args.numMotifs,1))) # size: [None,numMotifs]
		else: #no positional self.activities
			if self.args.trainStrandedActivities>0: # account for strand-specific activity biases
				if self.args.useEBound>0: 
					self.epBoundTensor   = tf.reduce_sum(tf.subtract(1.0,  pNotBoundTensor), reduction_indices=[1,2], name="epBoundTensor") # size: [None, numMotifs] #expected amount of binding
					self.epBoundTensorRC = tf.reduce_sum(tf.subtract(1.0,pNotBoundRCTensor), reduction_indices=[1,2], name="epBoundTensorRC") # size: [None, numMotifs] #expected amount of binding
				else:
					self.epBoundTensor = tf.subtract(1.0,tf.reduce_prod(pNotBoundTensor,reduction_indices=[1,2]), name="epBoundTensor") # numMotifs: [None, numMotifs] # p(bound)
					self.epBoundTensorRC = tf.subtract(1.0,tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2]), name="epBoundTensorRC") # size: [None, numMotifs] # p(bound)
				if self.args.potentiation>0:
					self.epBoundTensor = tf.transpose(tf.multiply(tf.transpose(self.epBoundTensor), self.seqPotentialTensor), name="epBoundTensor"); # [None, numMotifs]
					self.epBoundTensorRC = tf.transpose(tf.multiply(tf.transpose(self.epBoundTensorRC), self.seqPotentialTensor), name="epBoundTensorRC"); # [None, numMotifs]
					#print(tf.Tensor.get_shape(self.epBoundTensor))
					#print(tf.Tensor.get_shape(self.epBoundTensorRC))
				if self.args.bindingLimits>0:#note that when adding strand-specific self.activities when there are binding limits, the output will change without changing the params because you can't limit both simultaneously now.
					expectedActivityPerTF = tf.add( #min of positive self.activities and max of negative self.activities accounting for binding limits.
						tf.nn.relu(                        tf.minimum(tf.reshape(tf.multiply(bindingLimits,self.activities),(1,self.args.numMotifs)),tf.add(tf.multiply(self.epBoundTensor, self.activities), tf.multiply(self.epBoundTensorRC, activitiesRC)))), #positive
						tf.negative(tf.nn.relu(tf.negative(tf.maximum(tf.reshape(tf.multiply(bindingLimits,self.activities),(1,self.args.numMotifs)),tf.add(tf.multiply(self.epBoundTensor, self.activities), tf.multiply(self.epBoundTensorRC, activitiesRC)))))) #negative
						)#[None,numMotifs]
					#sys.stderr.write(" ".join([str(x) for x in tf.Tensor.get_shape(expectedActivity)])+"\n")
				else:
					expectedActivityPerTF = tf.add(tf.multiply(self.epBoundTensor, tf.reshape(self.activities,(self.args.numMotifs,1))), tf.multiply(self.epBoundTensorRC, tf.reshape(activitiesRC,(self.args.numMotifs,1)))) #[None,numMotifs]
			else: #no positional or strand effects
				if self.args.potentiation>0:
					self.epBoundTensor = tf.transpose(tf.multiply(tf.transpose(self.epBoundTensor),self.seqPotentialTensor), name="epBoundTensor"); # [None,numMotifs]
				if self.args.bindingLimits>0:
					self.epBoundTensor = tf.minimum(self.epBoundTensor, bindingLimits, name="epBoundTensor"); #[None, numMotifs]
				#expectedActivityPerTF = tf.multiply(self.epBoundTensor, tf.reshape(self.activities,(self.args.numMotifs,1))); #size: [None,numMotifs]
				expectedActivityPerTF = tf.multiply(self.epBoundTensor, self.activities); #size: [None,numMotifs]
		
		
		expectedActivity = tf.reduce_sum(expectedActivityPerTF, reduction_indices=[1])
		
		self.constant = tf.Variable(tf.zeros(1),name="constant")
		
		if self.args.accIsAct>0:
			accActivity = tf.Variable(tf.zeros(1),name="accessActiv")
			accActivELTensor = tf.multiply(self.seqPotentialTensor, accActivity) # [None,1]
			predELY= tf.add(tf.add(tf.reshape(expectedActivity, [-1]),tf.reshape(accActivELTensor, [-1])), self.constant, name="predELY") #size: [None]
		else:
			predELY= tf.add(tf.reshape(expectedActivity, [-1]),self.constant, name="predELY") #size: [None]
		self.realELY = tf.placeholder(tf.float32, [None]);
		
		EPSILON=0.0001
		
		self.mseTF = tf.reduce_mean(tf.square(self.realELY - predELY))
		if self.args.meanSDFile is not None:
			vals = np.loadtxt(self.args.meanSDFile, dtype = np.str);
			llMeans = vals[:,0].astype("float32");
			llSDs = vals[:,1].astype("float32");
			maxllMean = llMeans.max()
			llSDLen = llSDs.shape[0];
			if self.args.verbose>0: sys.stderr.write("Minimizing the negative log liklihood with an SD vector of length %i and a maximum value of %f\n"%(llSDLen,maxllMean))
			llSDTensor = tf.constant(llSDs, name="llSDs")
			predELLLIndeces = tf.minimum(llSDLen-1, tf.maximum(0,tf.to_int32(tf.round(tf.multiply(predELY,(llSDLen/maxllMean))))), name="predELLLInd") 
			predELYSDs = tf.nn.embedding_lookup(llSDTensor, predELLLIndeces, name="predELYSDs") #[None]
			predZ = tf.div(tf.subtract(predELY, self.realELY), predELYSDs, name="predZ")
			negLogLik = tf.reduce_sum(tf.square(predZ), name="negLogLik")
			self.myLoss = negLogLik;
		else:
			self.myLoss = self.mseTF;
		if self.args.L2 is not None and self.args.noTrainMotifs==0:
			if self.args.verbose>0: sys.stderr.write("Using L2 regularization of PWMs with lambda=%s\n"%(self.args.L2))
			self.args.L2 = float(self.args.L2);
			paramPenaltyL2Tensor = tf.nn.l2_loss(motifsTensor) #doesn't make sense to l2 loss the concentrations since this brings them to 0; however, bringing the PWM entries to 0 does make sense.
			self.myLoss = tf.add(self.myLoss, tf.multiply(paramPenaltyL2Tensor,self.args.L2));
		
		if self.args.L1 is not None:
			if self.args.verbose>0: sys.stderr.write("Using L1 regularization of self.activities with lambda=%s\n"%(self.args.L1))
			self.args.L1 = float(self.args.L1);
			paramPenaltyL1Tensor = tf.reduce_sum(tf.abs(self.activities))
			paramNumActivityTensor = tf.reduce_sum(tf.cast(tf.greater(tf.abs(self.activities),EPSILON),tf.int32))
			if self.args.potentiation>0:
				paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(self.potentiation))
				paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(self.potentiation),EPSILON),tf.int32))
			if self.args.trainStrandedActivities>0:
				paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(self.activityDiffs))
				paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(self.activityDiffs),EPSILON),tf.int32))
			if self.args.accIsAct>0:
				paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.abs(accActivity)
				paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(accActivity),EPSILON),tf.int32))
			if self.args.trainPositionalActivities>0:
				paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(tf.slice(self.positionalActivityBias,[0,0],[1,self.args.numMotifs]))) #penalize only the first column of positional self.activities
				#So only one position per TF is L1; the differences between the others are L2
				paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(tf.slice(self.positionalActivityBias,[0,0],[1,self.args.numMotifs])),EPSILON),tf.int32))
				if self.args.trainStrandedActivities>0:
					paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(tf.slice(self.positionalActivityBiasRC,[0,0],[1,self.args.numMotifs]))) 
					paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(tf.slice(self.positionalActivityBiasRC,[0,0],[1,self.args.numMotifs])),EPSILON),tf.int32))
			self.myLoss = tf.add(self.myLoss, tf.multiply(paramPenaltyL1Tensor,self.args.L1));
		
		if self.args.L2Pos is not None and self.args.trainPositionalActivities>0:
			self.args.L2Pos = float(self.args.L2Pos);
			paramPenaltyL2PosTensor = tf.reduce_sum(tf.abs(tf.subtract(tf.slice(self.positionalActivityBias,[0,0],[self.args.seqLen-1,self.args.numMotifs]),tf.slice(self.positionalActivityBias,[1,0],[self.args.seqLen-1,self.args.numMotifs]))))
			if self.args.trainStrandedActivities>0:
				paramPenaltyL2PosTensor = paramPenaltyL2PosTensor + tf.nn.l2_loss(tf.subtract(tf.slice(self.positionalActivityBiasRC,[0,0],[self.args.seqLen-1,self.args.numMotifs]),tf.slice(self.positionalActivityBiasRC,[1,0],[self.args.seqLen-1,self.args.numMotifs])))
			self.myLoss = tf.add(self.myLoss, tf.multiply(paramPenaltyL2PosTensor, self.args.L2Pos));
			
		
		self.global_step = tf.Variable(0, trainable=False, name="global_step")
		learningRate = self.args.learningRate
		if self.args.cycleLearningRate is not None:
			if self.args.verbose>0: sys.stderr.write("Cycling learning rate low,high,period %s\n"%self.args.cycleLearningRate)
			clrParams = self.args.cycleLearningRate.split(",")
			learningRate = TFHELP.cycle_learning_rate(float(clrParams[0]), float(clrParams[1]), self.global_step, int(clrParams[2]), name="learningRate")
		else:
			learningRate =tf.train.exponential_decay(learningRate, self.global_step, 175, self.args.learningRateED, staircase=True, name="learningRate")
		if self.args.useMomentumOptimizer > 0 or self.args.useRMSProp>0:
			if self.args.useMomentumOptimizer > 0 and self.args.useRMSProp>0:
				raise Exception("Cannot use both MomentumOptimizer and RMSProp! pick one!");
			if isinstance(self.args.momentum, basestring):
				momentum = tf.constant(float(self.args.momentum),name="momentum");
			else: 
				momentum=self.args.momentum
			if self.args.useMomentumOptimizer > 0:
				if self.args.verbose>0: sys.stderr.write("Using MomentumOptimizer instead of Adam\n")
				opt = tf.train.MomentumOptimizer(self.args.learningRate, momentum, use_nesterov = self.args.useNesterov>0);
		
			elif self.args.useRMSProp>0:
				if self.args.verbose>0: sys.stderr.write("Using RMSProp instead of Adam\n")
				opt = tf.train.RMSPropOptimizer(self.args.learningRate, decay = self.args.rmsDecay, momentum=momentum, epsilon=self.args.rmsEpsilon);
		else:
			if self.args.verbose>0: sys.stderr.write("Using AdamOptimizer\n")
			opt = tf.train.AdamOptimizer(self.args.learningRate, epsilon=self.args.Aepsilon, beta1=self.args.Abeta1, beta2=self.args.Abeta2);
		
		self.train_step = opt.minimize(self.myLoss, global_step=self.global_step);
		
		#raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
		if self.args.trace > 2:
			self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.args.threads-1, log_device_placement=True));
		else:
			self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.args.threads-1));
		
		
		if self.args.tensorboard is not None:
			#learning rate, loss
			tf.summary.scalar('learning rate', learningRate)
			tf.summary.scalar('MSE', self.mseTF)
			tf.summary.scalar('loss', self.myLoss)
			if self.args.trainPositionalActivities>0:
				tf.summary.histogram('TF positional activity bias', self.positionalActivityBias)
				if self.args.trainStrandedActivities>0:
					tf.summary.histogram('TF positional self.activities (RC)', self.positionalActivityBiasRC)
				if self.args.L2Pos is not None:
					tf.summary.scalar('positional L2 penalty', paramPenaltyL2PosTensor)
			if self.args.L1 is not None:
				tf.summary.scalar('general L1 penalty', paramPenaltyL1Tensor)
				tf.summary.scalar('number of non-zero L1 parameters', paramNumActivityTensor)
			# learned parameters
			tf.summary.histogram('TF self.activities', self.activities)
			if self.args.potentiation>0:
				tf.summary.histogram('TF potentiations', self.potentiation)
			if self.args.trainStrandedActivities>0:
				tf.summary.histogram('TF activity strand difference', self.activityDiffs)
			self.mergedSummaries = tf.summary.merge_all()
			self.summaryWriter = tf.summary.FileWriter(self.args.tensorboard, self.sess.graph)
		
		init_op = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		self.sess.run(init_op)
		if self.args.restore is not None:
			if self.args.verbose>1:
				sys.stderr.write("Loading initial parameter settings: %s\n"%(self.args.restore))
			reader = tf.train.NewCheckpointReader(self.args.restore);
			restoringThese = {v.name[0:-2]: v for v in tf.global_variables()};
			#print(list(restoringThese.keys()));
			if "concentrations" in reader.get_variable_to_shape_map().keys()  and  self.args.VARIABLE:
				raise Exception("BACKWARD COMPATIBILITY ERROR: loaded graph has variable named concentrations - re-run withOUT -VARIABLE option\n");
			if "Variable" in reader.get_variable_to_shape_map().keys() and not self.args.VARIABLE:
				raise Exception("BACKWARD COMPATIBILITY ERROR: re-run with -VARIABLE option\n");
				#restoringThese["Variable"] = restoringThese["concentrations"]
				#del restoringThese["concentrations"]
			restoringThese = {k: restoringThese[k] for k in list(restoringThese.keys()) if k in reader.get_variable_to_shape_map().keys()}
			import re;
			restoringThese = {k: restoringThese[k] for k in list(restoringThese.keys()) if re.search("global_step", k) is None}
			if self.args.clearAdamVars>0:
				restoringThese = {k: restoringThese[k] for k in list(restoringThese.keys()) if re.search("/Adam", k) is None}
				restoringThese = {k: restoringThese[k] for k in list(restoringThese.keys()) if re.search("beta[12]_power", k) is None}
			if self.args.verbose>1:
				sys.stderr.write("Loading these variables: %s\n"%(", ".join([k for k in list(restoringThese.keys())])))
			restoringThese = [restoringThese[k] for k in list(restoringThese.keys())]
			restorer = tf.train.Saver(restoringThese);
			restorer.restore(self.sess, self.args.restore)
		
		#make sure there are no unnamed variables
		import re;
		#if [k.name for k in tf.trainable_variables() if re.search("Variable", k.name) is not None].len() > 0:
		if len([k.name for k in tf.trainable_variables() if re.search("Variable", k.name) is not None]) > 0 and not self.args.VARIABLE:
			print([k.name for k in tf.trainable_variables()])
			raise(Exception("Error: one or more variables with default names: %s"%", ".join([k.name for k in tf.trainable_variables() if re.search("Variable", k.name) is not None])));
		
	def saveParams(self, sess):
		if self.args.noTrainMotifs==0:
			if (self.args.outFPre is None):
				outFile= sys.stdout;
			else:
				outFile = MYUTILS.smartGZOpen(self.args.outFPre+".pkdms",'w');
			#print out weight matrices
			pkdms = self.motifsTensor2.eval(session=self.sess); #[4,self.args.motifLen,1,self.args.numMotifs]);
			for i in range(0, self.args.numMotifs):
				outFile.write("#Motif %i\n"%(i));
				for j in range(0,4):
					outFile.write("#%s"%BASES[j]);
					for k in range(0,self.args.motifLen):
						outFile.write("\t%g"%(pkdms[j,k,0,i]));
					outFile.write("\n");
			outFile.write("\n");
			outFile.close();
		if self.args.trainPositionalActivities>0:
			if (self.args.outFPre is None):
				outFile= sys.stdout;
			else:
				outFile = MYUTILS.smartGZOpen(self.args.outFPre+".positional.gz",'w');
			outFile.write("TF\tposition\tpositionalActivityBias");
			positionalActivityBiasVals = self.positionalActivityBias.eval(session=self.sess).reshape((self.args.seqLen,self.args.numMotifs));
			if self.args.trainStrandedActivities>0:
				outFile.write("\tpositionalActivityBiasRC");
				positionalActivityBiasRCVals = self.positionalActivityBiasRC.eval(session=self.sess).reshape((self.args.seqLen,self.args.numMotifs));
			outFile.write("\n");
			for j in range(0,self.args.numMotifs):
				for i in range(0,self.args.seqLen):
					outFile.write("%i\t%i\t%g"%(j,i,positionalActivityBiasVals[i,j]));
					if self.args.trainStrandedActivities>0:
						outFile.write("\t%g"%positionalActivityBiasRCVals[i,j]);
					outFile.write("\n");
			outFile.close();
				
		if (self.args.outFPre is None):
			outFile= sys.stdout;
		else:
			outFile = MYUTILS.smartGZOpen(self.args.outFPre+".params",'w');
		#print params
		concs = self.logConcs.eval(session=self.sess).reshape((self.args.numMotifs));
		activityVals = self.activities.eval(session=self.sess).reshape((self.args.numMotifs));
		outFile.write("i\tlogConc\tactivity");
		if self.args.potentiation>0:
			outFile.write("\tpotentiations");
			potentiationVals = self.potentiation.eval(session=self.sess).reshape((self.args.numMotifs));
		if self.args.bindingLimits>0:
			outFile.write("\tbindingLimits");
			bindingLimitVals = bindingLimits.eval(session=self.sess).reshape((self.args.numMotifs));
		if self.args.trainStrandedActivities>0:
			outFile.write("\tactivityDiffs");
			activityDiffVals = self.activityDiffs.eval(session=self.sess).reshape((self.args.numMotifs));
		outFile.write("\n");
		if self.args.accIsAct>0:
			outFile.write("-1\t%g\t%g"%(accActivity.eval(session=self.sess),self.constant.eval(session=self.sess)));#intercepts
		else:
			outFile.write("-1\tNA\t%g"%self.constant.eval(session=self.sess));#intercepts
		if self.args.potentiation>0:
			outFile.write("\t%g"%(self.constantPot.eval(session=self.sess)));
		if self.args.bindingLimits>0:
			outFile.write("\tNA");
		if self.args.trainStrandedActivities>0:
			outFile.write("\tNA");
		outFile.write("\n");
		for i in range(0,self.args.numMotifs):
			outFile.write("%i\t%g\t%g"%(i, concs[i], activityVals[i]));#intercepts
			if self.args.potentiation>0:
				outFile.write("\t%g"%(potentiationVals[i]));
			if self.args.bindingLimits>0:
				outFile.write("\t%g"%(bindingLimitVals[i]));
			if self.args.trainStrandedActivities>0:
				outFile.write("\t%g"%(activityDiffVals[i]));
			outFile.write("\n");
		outFile.close();
	
