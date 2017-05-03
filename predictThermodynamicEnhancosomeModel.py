#!/usr/bin/env python
import warnings
import sys
import argparse
parser = argparse.ArgumentParser(description='Predicts biochemical transcription model on new data using OHC sequence input.')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file of one-hot-code sequences, preceeded by their expression value, tab delim', required=True);
parser.add_argument('-sl',dest='seqLen',	metavar='<seqLen>',help='Input sequence length in bp', required=True);
parser.add_argument('-b',dest='batch',	metavar='<batchSize>',help='Batch size to use to feed into the neural net [default = 1000] ', required=False, default = "1000");
parser.add_argument('-res',dest='restore',	metavar='<restoreFromCkptFile>',help='Restore NN from file ', required=True);
parser.add_argument('-nm',dest='numMotifs',	metavar='<numMotifs>',help='The number of motifs to include in the model [default=400]', required=False, default = "400");
parser.add_argument('-ml',dest='motifLen',	metavar='<motifLen>',help='Motif length [default = 20]', required=False, default = "20");
parser.add_argument('-dm',dest='motifsFP',	metavar='<defaultMotifFile>',help='Pre-load the PWMs with these', required=False);
parser.add_argument('-ic',dest='initConcs', metavar='<initConcs>',help='Initialize log concentrations to these (usually min Kds for the PWM; same order as others) [default log(concs)=0]', required=False);
parser.add_argument('-ia',dest='initActiv', metavar='<initActiv>',help='Initialize activities to these [defaults to gaussian]', required=False);
parser.add_argument('-ip',dest='initPotent', metavar='<initPotent>',help='Initialize potentiations to these [defaults to gaussian]', required=False);
parser.add_argument('-eb',dest='useEBound', action='count',help='use expected binding rather than prob binding?', required=False, default=0);
parser.add_argument('-po',dest='potentiation', action='count',help='add a layer of potentiation, modulating accessibility?', required=False, default=0);
parser.add_argument('-tftf',dest='interactions', metavar='<windowSize>',help='Add a layer of interaction filters to scale Kds based on nearby Kds within this window [default = do not use interaction terms]', required=False, default=0);
parser.add_argument('-posa',dest='trainPositionalActivities',action='count',help='Train slopes capturing a linear positional dependence? [default=no]. Note that this necessarily makes the model an EBound model after the potentiation step', required=False, default=0);
parser.add_argument('-stra',dest='trainStrandedActivities',action='count',help='Train activities capturing strand-specific differences in activity? [default=no]', required=False, default=0);
parser.add_argument('-ntc',dest='noTrainConcs', action='count',help='make TF concentrations constant (i.e. whatever concentrations are input)?', required=False, default=0);
parser.add_argument('-nta',dest='noTrainActivities', action='count',help='make TF activities constant (i.e. whatever activities are input)?', required=False, default=0);
parser.add_argument('-ntp',dest='noTrainPotentiations', action='count',help='make TF potentiations constant (i.e. whatever potentiations are input)?', required=False, default=0);
parser.add_argument('-ntm',dest='noTrainMotifs', action='count',help='Don\'t change the motif layer of variables?', required=False, default=0);
parser.add_argument('-t',dest='threads',	metavar='<threads>',help='Number of threads to make use of [default=1]',default = "1", required=False);
parser.add_argument('-o',dest='outFP', metavar='<outFile>',help='Where to output results [default=stdout]', required=False);
parser.add_argument('-ob',dest='outputBinding', action='count',help='Output binding score intermediates?', required=False, default=0);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);
#sys.argv = "predictThermodynamicEnhancosomeModel.py  -res ../../PBound_learned_params_pTpA_ACP_OHC.txt.ckpt -i 20160613_RefinedTSSs_Scer_20160203_R64_goodR_TSS-before-ATG_-190_to_-80.OHC.txt.gz -o test.txt  -v -v -v  -t 1 -b 1024  -ntm -po -nm 244 -sl 110".split();
#sys.argv = "predictThermodynamicEnhancosomeModel.py -i 20161024_average_promoter_ELs_per_seq_3p1E7_Gly_ALL.shuffled_OHC_test.txt.gz -res EBound_progressive_learning_pTpA_Gly.ACPM.ckpt -o test.txt -v -v -v -t 1 -b 1024 -sl 110 -nm 245 -ml 25 -po".split();
#sys.argv = "predictThermodynamicEnhancosomeModel.py -i 20161024_average_promoter_ELs_per_seq_3p1E7_Gly_ALL.shuffled_OHC_test.txt.gz -res EBound_progressive_learning_pTpA_Gly.A.ckpt -o test.txt -v -v -v -t 1 -b 1024 -sl 110 -nm 245 -ml 25 -ntm -ntc".split();
#sys.argv = "predictThermodynamicEnhancosomeModel.py -i ../../../20160525_NextSeq_pTpA_and_Abf1TATA/analysis/tensorflow/20160503_average_promoter_ELs_per_seq_atLeast100Counts.OHC.txt.gz -res EBound_progressive_learning_pTpA_Gly.A.ckpt -o test.txt -v -v -v -t 1 -b 1024 -sl 110 -nm 245 -ntc -ntm -ic /home/unix/cgdeboer/CIS-BP/YeTFaSCo/allTF_minKds_polyA_and_FZF1_justKds.txt".split();
args = parser.parse_args();
import MYUTILS
import PWM;
import tensorflow as tf
import numpy as np;

args.batch = int(args.batch);
args.seqLen = int(args.seqLen);
args.numMotifs = int(args.numMotifs);
args.motifLen = int(args.motifLen);
args.threads = int(args.threads);
args.interactions = int(args.interactions);

if (args.logFP is not None):
	logFile=MYUTILS.smartGZOpen(args.logFP,'w');
	sys.stderr=logFile;

if (args.outFP is not None):
	if args.verbose>0: sys.stderr.write("Outputting to file "+args.outFP+"*\n");
	outFile=MYUTILS.smartGZOpen(args.outFP,'w');
else:
	outFile = sys.stdout;


with open('/home/unix/cgdeboer/lib/python/SETUPOHCENHANCOSOMEMODEL.py') as fd:
	exec(fd.read())

#args.outFPre = 'test_params'; saveParams(sess);

#### set up tensorflow graph
##useful: tf.Tensor.get_shape(predELY)
##sequence layer
#BASES = ['A','C','G','T']
#ohcX = tf.placeholder(tf.float32, [None,4,args.seqLen,1]) # here, [None, 4,...] indicate that the first dimention is of unknown length
##motif parameters:
##allow for the user to provide a list of PWMs that we initialize the motifs tensor to
#if args.motifsFP is not None:
#	if args.verbose > 0: sys.stderr.write("Reading in default set of TF Motifs from %s\n"%(args.motifsFP));
#	#motifsFP is a one-per-line list of PWM paths
#	motifsFile=MYUTILS.smartGZOpen(args.motifsFP,'r');
#	defaultMotifs = np.random.normal(0,1,[4,args.motifLen,args.numMotifs])
#	i=0
#	for line in motifsFile:
#		if line is None or line == "" or line[0]=="#": continue
#		curPWM = PWM.loadPWM(line.rstrip());
#		pwmLen = curPWM.len();
#		pwmStart = 0;
#		pwmEnd = pwmLen; # actually end index+1
#		if pwmLen > args.motifLen: # can't fit the motif; 
#			#trim the terminal bases until pwmEnd ==args.motifLen; greedy
#			while (pwmEnd - pwmStart) > args.motifLen:
#				startInfo = 0.0;
#				endInfo = 0.0; 
#				for b in BASES:
#					startInfo += curPWM.mat[b][pwmStart]**2
#					endInfo += curPWM.mat[b][pwmEnd-1]**2
#				if startInfo > endInfo:
#					pwmEnd-=1;
#				else:
#					pwmStart+=1;
#		paddingFront = int((args.motifLen - (pwmEnd - pwmStart))/2); #truncated 
#		for bi in range(0,len(BASES)):
#			b=BASES[bi];
#			defaultMotifs[bi,:,i] = 0.0 # set other entries to 0
#			defaultMotifs[bi,paddingFront:(paddingFront+pwmEnd-pwmStart),i] = curPWM.mat[b][pwmStart:pwmEnd];
#		i+=1;
#	if args.noTrainMotifs>0:
#		motifsTensor = tf.constant(defaultMotifs.astype(np.float32),name="motifs") 
#	else:
#		motifsTensor = tf.Variable(tf.convert_to_tensor(defaultMotifs.astype(np.float32)),name="motifs") 
#else:
#	if args.noTrainMotifs>0:
#		motifsTensor = tf.constant(np.random.standard_normal([4,args.motifLen,args.numMotifs]).astype(np.float32),name="motifs") 
#	else:
#		motifsTensor = tf.Variable(tf.random_normal([4,args.motifLen,args.numMotifs]),name="motifs") 
#
#motifsRCTensor = tf.image.flip_left_right(tf.image.flip_up_down(motifsTensor)) #flips along first two dims; this works because order of bases is ACGT=rc(TGCA)
#
#motifsTensor2 = tf.reshape(motifsTensor, [4,args.motifLen,1,args.numMotifs]);#patch height, patch width, input channels, output channels
#motifsRCTensor2 = tf.reshape(motifsRCTensor, [4,args.motifLen,1,args.numMotifs]);
##output: [4, motifLen, 1, numMotifs]
#
############### INPUT ANY PROVIDED INITIAL PARAMS
### ACTIVITIES
#initActiv = np.random.standard_normal(args.numMotifs).astype(np.float32);
##input default concentrations if applicable
#if args.initActiv is not None:
#	vals = np.loadtxt(args.initActiv, dtype = np.str);
#	initActiv[0:vals.shape[0]] = vals[:,1].astype("float32");
#
#
#if args.noTrainActivities==0:
#	if args.verbose>0: sys.stderr.write("Training activities\n")
#	activities = tf.Variable(tf.convert_to_tensor(initActiv.reshape([args.numMotifs])),name="activities");
#else:
#	if args.verbose>0: sys.stderr.write("Not training activities\n")
#	activities = tf.constant(initActiv.reshape([args.numMotifs]),name="activities");
#
#if args.trainStrandedActivities>0:
#	if args.verbose>0: sys.stderr.write("Training stranded activities\n")
#	activityDiffs = tf.Variable(tf.zeros(args.numMotifs), name="activityDiffs")
#	activitiesRC = tf.add(activities, activityDiffs);
#
#if args.trainPositionalActivities>0:
#	if args.verbose>0: sys.stderr.write("Training positionally biased activities\n")
#	activitySlopes = tf.Variable(tf.zeros(args.numMotifs), name="activitySlopes"); #negative slopes mean more distant positions have higher activity
#	sequencePosition = tf.reshape(tf.constant(np.array(list(range(0,args.seqLen))).astype("float32")), shape = (args.seqLen,1)) # so "activity" parameter is at the most distant position
#	positionalActivity = tf.add(tf.matmul(sequencePosition, tf.reshape(activitySlopes, shape = (1,args.numMotifs))), activities) #[seqLen, numMotifs]
#	if args.trainStrandedActivities:
#		activitySlopesDiff = tf.Variable(tf.zeros(args.numMotifs), name="activitySlopesDiff"); 
#		activitySlopesRC = tf.add(activitySlopes, activitySlopesDiff); #[numMotifs]
#		positionalActivityRC = tf.add(tf.matmul(sequencePosition, tf.reshape(activitySlopesRC, shape = (1,args.numMotifs))), activitiesRC) #[seqLen, numMotifs]
#
### CONCENTRATIONS
#initConcs = np.zeros(args.numMotifs).astype(np.float32);
##input default concentrations if applicable
#if args.initConcs is not None:
#	vals = np.loadtxt(args.initConcs, dtype = np.str);
#	initConcs[0:vals.shape[0]] = vals[:,1].astype("float32");
#
#if args.noTrainConcs==0:
#	if args.verbose>0: sys.stderr.write("Training concentrations\n")
#	logConcs = tf.Variable(tf.convert_to_tensor(initConcs.reshape((1,1,1,args.numMotifs)))); ###TODO name="concentrations"
#else:
#	if args.verbose>0: sys.stderr.write("Not training concentrations\n")
#	logConcs = tf.constant(initConcs.reshape((1,1,1,args.numMotifs)));
#
##motif layer: conv layer 1
#	#nodes: motifs * orientations * positions
#	#params: motifs * motifLens * 4
##strides all =1 - this is the step size
## zero padding =SAME makes output dims =input; valid does not pad with 0s
##motifScanTensor = tf.nn.conv2d(ohcX, motifsTensor2, strides = [1,1,1,1], padding='VALID', name="motifScan") #VALID so that the output dimensions are 1 * seqLen-motifLen+1, ...
##motifScanRCTensor= tf.nn.conv2d(ohcX,motifsRCTensor2, strides = [1,1,1,1], padding='VALID', name="motifScanRC") #VALID so that the output dimensions are 1 * seqLen-motifLen+1, ...
##### ##outputs [None,1,seqLen-motifLen+1,numMotifs]
##these are log(Kds)
#motifScanTensor = tf.nn.conv2d(ohcX, motifsTensor2, strides = [1,4,1,1], padding='SAME', name="motifScan") 
#motifScanRCTensor= tf.nn.conv2d(ohcX,motifsRCTensor2, strides = [1,4,1,1], padding='SAME', name="motifScanRC") 
###outputs [None,1,seqLen,numMotifs]
#
#if args.interactions>0:
#	#the following code implements TF cooperativity/competition
#	#learned coeficients act as a scale of the log(kd) using nearby Kds for other (and the same) factors. For instance, for Kd_i_p of factor i at position p, log(Kdcoop_i_p) = log(Kd_i_p) + sum[all nearby Kds at distance d for factor j] ( C_i_j_d * Kd_j_(p+d) )
#	#these are the kd cooperativity filters
#	#here, first F/R refers to current motif [dim 3], second F/R refers to interacting motif [0];
#	#for these parameters, Ccoop<0 implies competition, Ccoop>0 implies cooperation
#	kdBonusInteractionsFilterFF = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsFF") 
#	kdBonusInteractionsFilterFR = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsFR") 
#	kdBonusInteractionsFilterRF = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsRF") 
#	kdBonusInteractionsFilterRR = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsRR") 
#	#reshape motifScanTensor from [None, 1, seqLen, numMotifs] to [None, numMot, seqLen, 1]
#	motifScanTensorReshape = tf.transpose(motifScanTensor, perm = [0,3,2,1]); # apply filter to these existing Kds # [None, numMot, seqLen, 1]
#	motifScanRCTensorReshape = tf.transpose(motifScanRCTensor, perm = [0,3,2,1]);
#	#apply filters
#	#Each of these are [None, 1, seqLen, numMot]
#	kdBonusScanTensorFF = tf.nn.conv2d(motifScanTensorReshape, kdBonusInteractionsFilterFF, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanFF") # interactions between current F motif [dim=3] and all other F motifs
#	kdBonusScanTensorFR = tf.nn.conv2d(motifScanRCTensorReshape, kdBonusInteractionsFilterFR, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanFR") # interaction between current F motif and all other R motifs
#	kdBonusScanTensorRF = tf.nn.conv2d(motifScanTensorReshape, kdBonusInteractionsFilterRF, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanRF") # interactions between current R motif [dim=3] and all other F motifs
#	kdBonusScanTensorRR = tf.nn.conv2d(motifScanRCTensorReshape, kdBonusInteractionsFilterRR, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanRR") # interaction between current R motif and all other R motifs
#	#add the bonus Kds from neighboring motifs to original motifs
#	motifScanTensorWithBonus = tf.add(tf.add(kdBonusScanTensorFF, kdBonusScanTensorFR), motifScanTensor); #log(Kdcoop) = log(Kd) + log(Kdbonus) [None, 1, seqLen, numMot]
#	motifScanRCTensorWithBonus = tf.add(tf.add(kdBonusScanTensorRF, kdBonusScanTensorRR), motifScanRCTensor); 
#	logKdConcRatioTensor = tf.sub(logConcs,motifScanTensorWithBonus) # [None, 1, seqLen,numMotifs] 
#	logKdConcRatioRCTensor = tf.sub(logConcs,motifScanRCTensorWithBonus) # [None, 1, seqLen,numMotifs] 
#else:
#	logKdConcRatioTensor = tf.sub(logConcs,motifScanTensor) # [None, 1, seqLen,numMotifs] 
#	logKdConcRatioRCTensor = tf.sub(logConcs,motifScanRCTensor) # [None, 1, seqLen,numMotifs] 
#
#
#pNotBoundTensor = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioTensor))); # size: [None,1,seqLen,numMotifs]
#pNotBoundRCTensor = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioRCTensor))); # size: [None,1,seqLen,numMotifs]
#
#if args.useEBound>0: 
#	if args.verbose>0: sys.stderr.write("Using E-bound\n")
#	epBoundTensor = tf.add(tf.reduce_sum(tf.sub(1.0,pNotBoundRCTensor), reduction_indices=[1,2]),tf.reduce_sum(tf.sub(1.0,pNotBoundTensor), reduction_indices=[1,2])) # size: [None, numMotifs] #expected amount of binding
#else:
#	if args.verbose>0: sys.stderr.write("Using P-bound\n")
#	epBoundTensor = tf.sub(1.0,tf.mul(tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2]),tf.reduce_prod(pNotBoundTensor, reduction_indices=[1,2]))) # size: [None, numMotifs] # p(bound)
#
### POTENTIATION
#if args.potentiation>0:
#	if args.verbose>0: sys.stderr.write("Using potentiation layer\n")
#	initPotent = np.random.standard_normal(args.numMotifs).astype(np.float32);
#	#input default concentrations if applicable
#	if args.initPotent is not None:
#		vals = np.loadtxt(args.initPotent, dtype = np.str);
#		initPotent[0:vals.shape[0]] = vals[:,1].astype("float32");
#	if args.noTrainPotentiations==0:
#		if args.verbose>0: sys.stderr.write("Training potentiations\n")
#		potentiation = tf.Variable(tf.convert_to_tensor(initPotent.reshape([args.numMotifs])),name="potents");
#	else:
#		if args.verbose>0: sys.stderr.write("Not training potentiations\n")
#		potentiation = tf.constant(initPotent.reshape([args.numMotifs]),name="potents");
#	seqPotentialByTFTensor = tf.mul(epBoundTensor, potentiation); #size: [None,numMotifs]
#	constantPot = tf.Variable(tf.zeros(1),name="constantPot")
#	seqPotentialTensor = tf.sigmoid(tf.add(tf.reduce_sum(seqPotentialByTFTensor,reduction_indices=[1]), constantPot)) #[None, 1]
#else:
#	if args.verbose>0: sys.stderr.write("Not using potentiation layer\n")
#
#if args.trainPositionalActivities>0: # account for positional activity with linear scaling of activity
#	if args.trainStrandedActivities>0: # account for strand-specific activity biases
#		pBoundPerPos = tf.sub(1.0,pNotBoundTensor) # size: [None,1,seqLen,numMotifs]
#		pBoundPerPosRC = tf.sub(1.0,pNotBoundRCTensor) # size: [None,1,seqLen,numMotifs]
#		if args.potentiation>0:
#			pBoundPerPos = tf.transpose(tf.mul(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
#			pBoundPerPosRC = tf.transpose(tf.mul(tf.transpose(pBoundPerPosRC, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
#		#print(tf.Tensor.get_shape(pBoundPerPos))
#		#print(tf.Tensor.get_shape(positionalActivity))
#		expectedActivity = tf.matmul(tf.reshape(pBoundPerPos, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivity, (args.seqLen*args.numMotifs,1))) # size: [None,1]
#		expectedActivityRC = tf.matmul(tf.reshape(pBoundPerPosRC, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivityRC, (args.seqLen*args.numMotifs,1))) # size: [None,1]
#		expectedActivity = tf.add(expectedActivity, expectedActivityRC);
#	else:
#		if args.useEBound>0:
#			pBoundPerPos = tf.add(tf.sub(1.0,pNotBoundTensor), tf.sub(1.0,pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
#		else:
#			pBoundPerPos = tf.sub(1.0,tf.mul(pNotBoundTensor, pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
#		#print(tf.Tensor.get_shape(pBoundPerPos))
#		if args.potentiation>0:
#			pBoundPerPos = tf.transpose(tf.mul(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
#		#print(tf.Tensor.get_shape(pBoundPerPos))
#		#print(tf.Tensor.get_shape(positionalActivity))
#		expectedActivity = tf.matmul(tf.reshape(pBoundPerPos, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivity, (args.seqLen*args.numMotifs,1))) # size: [None,1]
#else: #no positional activities
#	if args.trainStrandedActivities>0: # account for strand-specific activity biases
#		if args.useEBound>0: 
#			epBoundTensor = tf.reduce_sum(tf.sub(1.0,pNotBoundTensor), reduction_indices=[1,2]) # size: [None, numMotifs] #expected amount of binding
#			epBoundTensorRC = tf.reduce_sum(tf.sub(1.0,pNotBoundRCTensor), reduction_indices=[1,2]) # size: [None, numMotifs] #expected amount of binding
#		else:
#			epBoundTensor = tf.sub(1.0,tf.reduce_prod(pNotBoundTensor,reduction_indices=[1,2])) # size: [None, numMotifs] # p(bound)
#			epBoundTensorRC = tf.sub(1.0,tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2])) # size: [None, numMotifs] # p(bound)
#		if args.potentiation>0:
#			epBoundTensor = tf.transpose(tf.mul(tf.transpose(epBoundTensor), seqPotentialTensor)); # [None, numMotifs]
#			epBoundTensorRC = tf.transpose(tf.mul(tf.transpose(epBoundTensorRC), seqPotentialTensor)); # [None, numMotifs]
#			#print(tf.Tensor.get_shape(epBoundTensor))
#			#print(tf.Tensor.get_shape(epBoundTensorRC))
#		expectedActivity = tf.add(tf.matmul(epBoundTensor, tf.reshape(activities,(args.numMotifs,1))), tf.matmul(epBoundTensorRC, tf.reshape(activitiesRC,(args.numMotifs,1)))) #[None,1]
#	else: #no positional or strand effects
#		if args.potentiation>0:
#			epBoundTensor = tf.transpose(tf.mul(tf.transpose(epBoundTensor),seqPotentialTensor)); # [None,numMotifs]
#		expectedActivity = tf.matmul(epBoundTensor, tf.reshape(activities,(args.numMotifs,1))); #size: [None,1]
#
#constant = tf.Variable(tf.zeros(1),name="constant")
#
#predELY= tf.add(tf.reshape(expectedActivity, [-1]),constant) #size: [None]
#
#
##raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
#sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads));
#init_op = tf.initialize_all_variables()
#saver = tf.train.Saver()
#sess.run(init_op)
#if args.restore is not None:
#	if args.verbose>1:
#		sys.stderr.write("Loading initial parameter settings: %s\n"%(args.restore))
#	reader = tf.train.NewCheckpointReader(args.restore);
#	if args.verbose>1:
#		sys.stderr.write("Loading these variables: %s\n"%(", ".join([k.name for k in tf.all_variables() if k.name in ["%s:0"%s for s in reader.get_variable_to_shape_map().keys()]])))
#	restorer = tf.train.Saver([k for k in tf.all_variables() if k.name in ["%s:0"%s for s in reader.get_variable_to_shape_map().keys()]]);
#	restorer.restore(sess, args.restore)
#
outFile.write("actual\tpredicted");
if args.potentiation>0:
	outFile.write("\tpred_openness");

if args.outputBinding > 0:
	if args.trainPositionalActivities > 0:
		raise Exception("Cannot output binding values while using positional activities");
	outFile.write("\t" + "\t".join(["Binding_%i"%x for x in range(0,args.numMotifs)]))
	if args.trainStrandedActivities>0:
		outFile.write("\t" + "\t".join(["RCBinding_%i"%x for x in range(0,args.numMotifs)]))

outFile.write("\n");

b = 0
batchX = np.zeros((args.batch,4,args.seqLen,1))
batchY = np.zeros((args.batch))
inFile=MYUTILS.smartGZOpen(args.inFP,'r');
for line in inFile:
	if line is None or line == "" or line[0]=="#": continue
	curData = np.fromstring(line, dtype=float, sep="\t")
	batchY[b]=curData[0];
	batchX[b,:,:,0] = curData[1:].reshape((4,args.seqLen))
	b+=1
	if b==args.batch:
		curPredY = predELY.eval(session=sess, feed_dict={ohcX: batchX})
		if args.outputBinding>0:
			bindingAmount = epBoundTensor.eval(session=sess, feed_dict={ohcX: batchX});
			if args.trainStrandedActivities>0:
				bindingAmountRC = epBoundTensorRC.eval(session=sess, feed_dict={ohcX: batchX});
		if args.potentiation>0:
			curPredOpenness = seqPotentialTensor.eval(session=sess, feed_dict={ohcX: batchX})
		for i in range(0,batchY.shape[0]):
			outFile.write("%g\t%g"%(batchY[i],curPredY[i]));
			if args.potentiation>0:
				outFile.write("\t%g"%(curPredOpenness[i]));
			if args.outputBinding > 0:
				outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmount[i,]])));
				if args.trainStrandedActivities>0:
					outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmountRC[i,]])));
			outFile.write("\n");
		b=0;

inFile.close();

#test with remaining data, but remove anything past b
if b > 0:
	batchX = batchX[0:b,:,:,:]
	batchY = batchY[0:b];
	curPredY = predELY.eval(session=sess, feed_dict={ohcX: batchX})
	if args.outputBinding>0:
		bindingAmount = epBoundTensor.eval(session=sess, feed_dict={ohcX: batchX});
		if args.trainStrandedActivities>0:
			bindingAmountRC = epBoundTensorRC.eval(session=sess, feed_dict={ohcX: batchX});
	if args.potentiation>0:
		curPredOpenness = seqPotentialTensor.eval(session=sess, feed_dict={ohcX: batchX})
	for i in range(0,batchY.shape[0]):
		outFile.write("%g\t%g"%(batchY[i],curPredY[i]));
		if args.potentiation>0:
			outFile.write("\t%g"%(curPredOpenness[i]));
		if args.outputBinding > 0:
			outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmount[i,]])));
			if args.trainStrandedActivities>0:
				outFile.write("\t%s"%("\t".join(["%g"%ba for ba in bindingAmountRC[i,]])));
		outFile.write("\n");

sess.close()
if (args.logFP is not None):
	logFile.close();

