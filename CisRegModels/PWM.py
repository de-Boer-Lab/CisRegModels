import math
import random
import copy
BASES = ['A','T','G','C']
UNIFORM = {'A':0.25,'T':0.25,'G':0.25,'C':0.25};
def loadPWM(fileName):
	inFile = open(fileName,'r')
	mat={};
	myLen=-1;
	count=0
	for line in inFile:
		if line is None or line == "" or line[0]=="#": continue
		data=line.rstrip().split("\t");
		curBase=BASES[count];
		if data[0] in BASES:
			curBase=data[0];
			data=data[1:]
		data=[float(d) for d in data]
		mat[curBase]=data
		if myLen!=-1 and myLen !=len(data):
			raise Exception('PWM does not have the same number of entries per row for %s' % (fileName));
		myLen=len(data)
		count+=1;
	inFile.close();
	if count!=4:
		raise Exception('PWM has the wrong number of rows for %s; count=%d' % (filename,count))
	return PWM(mat)
class PWM(object):
	def __init__(self,mat):
		self.mat=mat
		self.isPKdM=False;
		self.myRC=None;
	def deepcopy(self):
		theCopy = PWM(copy.deepcopy(self.mat))
		theCopy.isPKdM = self.isPKdM;
		return theCopy
	def to_PWM(self,prior=UNIFORM,pseudocount=0.001):
		newPWM = self.deepcopy();
		newPWM.to_PWM_B(prior,pseudocount)
		return newPWM;
	def to_PWM_B(self,prior=UNIFORM,pseudocount=0.001):
		self.normalize_B()
		for b in BASES:
			for i in range(0,(self.len())):
				self.mat[b][i] += pseudocount;
		self.normalize_B();
		for b in BASES:
			for i in range(0,(self.len())):
				self.mat[b][i] = math.log(self.mat[b][i]/prior[b],2)
	def getMin(self):
		score = 0.0;
		seq = ""
		for i in range(0,(self.len())):
			curMin = self.mat["A"][i]
			curBase = "A"
			for b in BASES:
				if self.mat[b][i] < curMin:
					curBase = b;
					curMin = self.mat[b][i]
			seq = seq  +curBase;
			score = score + curMin;
		return (score, seq);
	def getHitSeqs(self, thresholdScore):
		(bestScore, bestSeq) = self.getMax();
		return self.getAlternateBSsMeetingThreshold(bestScore, bestSeq, thresholdScore, 0)
	def getAlternateBSsMeetingThreshold(self, curScore, curSeq, thresholdScore, baseI):
		if baseI>=self.len():
			return [curSeq];
		curSeqs = [];
		for b in BASES:
			testScore = curScore + self.mat[b][baseI] - self.mat[curSeq[baseI]][baseI];
			if testScore>=thresholdScore:
				curSeqs = curSeqs + self.getAlternateBSsMeetingThreshold(testScore, curSeq[0:baseI]+b+curSeq[(baseI+1):len(curSeq)], thresholdScore, baseI+1)
		return curSeqs;
		
	def getIC(self,prior=UNIFORM,pseudocount=0.001):
		IC = 0.0;
		for i in range(0,(self.len())):
			for b in BASES:
				IC+= self.mat[b][i] * math.log((self.mat[b][i]+pseudocount)/prior[b], 2)
		return IC
	def getMax(self):
		score = 0.0;
		seq = ""
		for i in range(0,(self.len())):
			curMax = self.mat["A"][i]
			curBase = "A"
			for b in BASES:
				if self.mat[b][i] > curMax:
					curBase = b;
					curMax = self.mat[b][i]
			seq = seq  +curBase;
			score = score + curMax;
		return (score, seq);
	def to_PKdM(self,prior=UNIFORM,pseudocount=0.001):
		newPWM = self.deepcopy();
		newPWM.to_PKdM_B(prior,pseudocount)
		return newPWM;
	def to_PKdM_B(self,prior=UNIFORM,pseudocount=0.001):
		self.normalize_B()
		self.isPKdM=True;
		for b in BASES:
			for i in range(0,(self.len())):
				self.mat[b][i] += pseudocount;
		self.normalize_B();
		for b in BASES:
			for i in range(0,(self.len())):
				self.mat[b][i] = math.log(prior[b]/self.mat[b][i],2)
	def to_PFM_from_PKDM_B(self,prior=UNIFORM):
		for b in BASES:
			for i in range(0,(self.len())):
				self.mat[b][i] = prior[b]/(2**self.mat[b][i]) #math.log(prior[b]/self.mat[b][i],2)
		self.normalize_B();
		self.isPKdM = False;
	def len(self):
		return len(self.mat['A']);
	def normalize_B(self):
		sum = [0.0]*self.len()
		for b in BASES:
			for i in range(0,(self.len())):
				sum[i] += self.mat[b][i]
		for b in BASES:
			for i in range(0,(self.len())):
				self.mat[b][i] = self.mat[b][i]/sum[i]
	def generateBS(self):
		curSeq = "";
		for i in range(0, self.len()):
			curRand = random.random();
			for b in BASES:
				curRand = curRand - self.mat[b][i];
				if curRand < 0:
					curSeq=curSeq + b;
					break;
		return curSeq;
	def revcomp(self):
		newPWM = self.deepcopy();
		newPWM.revcomp_B()
		return newPWM;
	def addNToMat_B(self,use=max):
		self.mat['N']=copy.deepcopy(self.mat['A']);
		for i in range(0, self.len()):
			for b in BASES:
				self.mat['N'][i]=use(self.mat[b][i],self.mat['N'][i])
	def negate_B(self):
		self.mat['G']=[-x for x in self.mat['G']];
		self.mat['C']=[-x for x in self.mat['C']];
		self.mat['A']=[-x for x in self.mat['A']];
		self.mat['T']=[-x for x in self.mat['T']];
	def revcomp_B(self):
		temp = list(reversed(self.mat['G']));
		self.mat['G']=list(reversed(self.mat['C']));
		self.mat['C']=temp
		temp = list(reversed(self.mat['A']));
		self.mat['A']=list(reversed(self.mat['T']));
		self.mat['T']=temp
	def dsScan(self,seq):
		if self.myRC is None:
			self.myRC = self.revcomp();
		return self.scan(seq), self.myRC.scan(seq);
	def dsGomerPBounds(self, seq, conc, kdScale=0):
		if not self.isPKdM:
			raise Exception("Trying to run gomerScore on a non-PKdM");
		top, bottom = self.dsScan(seq); #returns log kds
		top = [1.0 - (1.0/(1.0 + (math.exp((conc-kd)*math.exp(kdScale))))) for kd in top]
		bottom = [1.0 - (1.0/(1.0 + (math.exp((conc-kd)*math.exp(kdScale))))) for kd in bottom]
		#top = [1.0 - (1.0/(1.0 + (math.exp(conc-kd))**math.exp(kdScale))) for kd in top]
		#bottom = [1.0 - (1.0/(1.0 + (math.exp(conc-kd))**math.exp(kdScale))) for kd in bottom]
		return [top, bottom]
	def gomerScore(self,seq, conc):
		if self.myRC is None:
			self.myRC = self.revcomp();
		top = self.gomerScore_SS(seq, conc);
		bottom = self.myRC.gomerScore_SS(seq, conc);
		#print("scanning %s; top = %g; bottom = %g; combined = %g"%(seq,top,bottom, 1.0 - (1.0 - top)*(1.0-bottom)));
		return 1.0 - (1.0 - top)*(1.0-bottom);
	def gomerScore_SS(self,seq, conc):
		if not self.isPKdM:
			raise Exception("Trying to run gomerScore on a non-PKdM");
		logPNotBound = 0.0;
		for i in range(0,(len(seq)-self.len()+1)):
			kdi=0.0
			for j in range(0,(self.len())):
				if seq[i+j] in self.mat:
					kdi+=self.mat[seq[i+j]][j]
			logPNotBound-= math.log(1 + math.exp(conc-kdi),2)
		return 1.0 - 2.0**logPNotBound;
	def scan(self,seq):
		allScores = [0.0] * (len(seq)-self.len()+1);
		for i in range(0,(len(seq)-self.len()+1)):
			for j in range(0,(self.len())):
				allScores[i]+=self.mat[seq[i+j]][j]
		return allScores
	def output(self,outStream):
		outStream.write(self.to_s())
	def to_s(self):
		pwmStr="";
		for b in self.mat:
			pwmStr +=b+"\t"+"\t".join([str(e) for e in self.mat[b]])+"\n"
		return pwmStr
	def to_REDUCE(self):
		pwmStr = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n<matrix_reduce>\n<psam_length>%i</psam_length>\n<psam>\n"%(self.len());
		baseOrder = ["A","C","G","T"];
		pwmStr = pwmStr +"#"+ "\t".join(baseOrder)+"\n"
		for i in range(0, self.len()):
			for b in baseOrder:
				pwmStr = pwmStr + str(self.mat[b][i])+"\t"
			pwmStr=pwmStr + "\n"
		return pwmStr + "</psam>\n</matrix_reduce>\n";
