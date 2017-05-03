
if not hasattr(args, 'L2'):
	args.L2=0.001;
if not hasattr(args, 'L1'):
	args.L1=0.001;
if not hasattr(args, 'learningRate'):
	args.learningRate=0.001;
if not hasattr(args, 'meanSDFile'):
	args.meanSDFile=None;

### set up tensorflow graph
#useful: tf.Tensor.get_shape(predELY)
#sequence layer
BASES = ['A','C','G','T']
ohcX = tf.placeholder(tf.float32, [None,4,args.seqLen,1]) # here, [None, 4,...] indicate that the first dimention is of unknown length
#motif parameters:
#allow for the user to provide a list of PWMs that we initialize the motifs tensor to
if args.motifsFP is not None:
	if args.verbose > 0: sys.stderr.write("Reading in default set of TF Motifs from %s\n"%(args.motifsFP));
	#motifsFP is a one-per-line list of PWM paths
	motifsFile=MYUTILS.smartGZOpen(args.motifsFP,'r');
	defaultMotifs = np.random.normal(0,1,[4,args.motifLen,args.numMotifs])
	i=0
	for line in motifsFile:
		if line is None or line == "" or line[0]=="#": continue
		curPWM = PWM.loadPWM(line.rstrip());
		pwmLen = curPWM.len();
		pwmStart = 0;
		pwmEnd = pwmLen; # actually end index+1
		if pwmLen > args.motifLen: # can't fit the motif; 
			#trim the terminal bases until pwmEnd ==args.motifLen; greedy
			while (pwmEnd - pwmStart) > args.motifLen:
				startInfo = 0.0;
				endInfo = 0.0; 
				for b in BASES:
					startInfo += curPWM.mat[b][pwmStart]**2
					endInfo += curPWM.mat[b][pwmEnd-1]**2
				if startInfo > endInfo:
					pwmEnd-=1;
				else:
					pwmStart+=1;
		paddingFront = int((args.motifLen - (pwmEnd - pwmStart))/2); #truncated 
		for bi in range(0,len(BASES)):
			b=BASES[bi];
			defaultMotifs[bi,:,i] = 0.0 # set other entries to 0
			defaultMotifs[bi,paddingFront:(paddingFront+pwmEnd-pwmStart),i] = curPWM.mat[b][pwmStart:pwmEnd];
		i+=1;
	if args.noTrainMotifs>0:
		motifsTensor = tf.constant(defaultMotifs.astype(np.float32),name="motifs") 
	else:
		motifsTensor = tf.Variable(tf.convert_to_tensor(defaultMotifs.astype(np.float32)),name="motifs") 
else:
	if args.noTrainMotifs>0:
		motifsTensor = tf.constant(np.random.standard_normal([4,args.motifLen,args.numMotifs]).astype(np.float32),name="motifs") 
	else:
		motifsTensor = tf.Variable(tf.random_normal([4,args.motifLen,args.numMotifs]),name="motifs") 

motifsRCTensor = tf.image.flip_left_right(tf.image.flip_up_down(motifsTensor)) #flips along first two dims; this works because order of bases is ACGT=rc(TGCA)

motifsTensor2 = tf.reshape(motifsTensor, [4,args.motifLen,1,args.numMotifs]);#patch height, patch width, input channels, output channels
motifsRCTensor2 = tf.reshape(motifsRCTensor, [4,args.motifLen,1,args.numMotifs]);
#output: [4, motifLen, 1, numMotifs]

############## INPUT ANY PROVIDED INITIAL PARAMS
## ACTIVITIES
initActiv = np.random.standard_normal(args.numMotifs).astype(np.float32);
#input default concentrations if applicable
if args.initActiv is not None:
	vals = np.loadtxt(args.initActiv, dtype = np.str);
	initActiv[0:vals.shape[0]] = vals[:,1].astype("float32");


if args.noTrainActivities==0:
	if args.verbose>0: sys.stderr.write("Training activities\n")
	activities = tf.Variable(tf.convert_to_tensor(initActiv.reshape([args.numMotifs])),name="activities");
else:
	if args.verbose>0: sys.stderr.write("Not training activities\n")
	activities = tf.constant(initActiv.reshape([args.numMotifs]),name="activities");

if args.trainStrandedActivities>0:
	if args.verbose>0: sys.stderr.write("Training stranded activities\n")
	activityDiffs = tf.Variable(tf.zeros(args.numMotifs), name="activityDiffs")
	activitiesRC = tf.add(activities, activityDiffs);

if args.trainPositionalActivities>0:
	if args.verbose>0: sys.stderr.write("Training positionally biased activities\n")
	activitySlopes = tf.Variable(tf.zeros(args.numMotifs), name="activitySlopes"); #negative slopes mean more distant positions have higher activity
	sequencePosition = tf.reshape(tf.constant(np.array(list(range(0,args.seqLen))).astype("float32")), shape = (args.seqLen,1)) # so "activity" parameter is at the most distant position
	positionalActivity = tf.add(tf.matmul(sequencePosition, tf.reshape(activitySlopes, shape = (1,args.numMotifs))), activities) #[seqLen, numMotifs]
	if args.trainStrandedActivities:
		activitySlopesDiff = tf.Variable(tf.zeros(args.numMotifs), name="activitySlopesDiff"); 
		activitySlopesRC = tf.add(activitySlopes, activitySlopesDiff); #[numMotifs]
		positionalActivityRC = tf.add(tf.matmul(sequencePosition, tf.reshape(activitySlopesRC, shape = (1,args.numMotifs))), activitiesRC) #[seqLen, numMotifs]

## CONCENTRATIONS
initConcs = np.zeros(args.numMotifs).astype(np.float32);
#input default concentrations if applicable
if args.initConcs is not None:
	vals = np.loadtxt(args.initConcs, dtype = np.str);
	initConcs[0:vals.shape[0]] = vals[:,1].astype("float32");

if args.noTrainConcs==0:
	if args.verbose>0: sys.stderr.write("Training concentrations\n")
	logConcs = tf.Variable(tf.convert_to_tensor(initConcs.reshape((1,1,1,args.numMotifs)))); ###TODO name="concentrations" #note that these changes will lead to incompatibility
else:
	if args.verbose>0: sys.stderr.write("Not training concentrations\n")
	logConcs = tf.constant(initConcs.reshape((1,1,1,args.numMotifs))); ###TODO name="concentrations"

## binding limits
if args.bindingLimits:
	initBL = np.ones(args.numMotifs).astype(np.float32);
	#input default concentrations if applicable
	if args.initBindLim is not None:
		vals = np.loadtxt(args.initBindLim, dtype = np.str);
		initBL[0:vals.shape[0]] = np.log(vals[:,1].astype("float32"));  #thus they are input as non-log

	if args.noTrainBL==0:
		if args.verbose>0: sys.stderr.write("Training binding limits\n")
		logBindingLimits = tf.Variable(tf.convert_to_tensor(initBL.reshape((args.numMotifs))), name="logBindingLimits");  #[numMotifs]
	else:
		if args.verbose>0: sys.stderr.write("Not training binding limits\n")
		logBindingLimits = tf.constant(initBL.reshape((args.numMotifs)), name="logBindingLimits");  #[numMotifs]
	bindingLimits=tf.exp(logBindingLimits, name="bindingLimits");

#motif layer: conv layer 1
	#nodes: motifs * orientations * positions
	#params: motifs * motifLens * 4
#strides all =1 - this is the step size
# zero padding =SAME makes output dims =input; valid does not pad with 0s
#motifScanTensor = tf.nn.conv2d(ohcX, motifsTensor2, strides = [1,1,1,1], padding='VALID', name="motifScan") #VALID so that the output dimensions are 1 * seqLen-motifLen+1, ...
#motifScanRCTensor= tf.nn.conv2d(ohcX,motifsRCTensor2, strides = [1,1,1,1], padding='VALID', name="motifScanRC") #VALID so that the output dimensions are 1 * seqLen-motifLen+1, ...
#### ##outputs [None,1,seqLen-motifLen+1,numMotifs]
#these are log(Kds)
motifScanTensor = tf.nn.conv2d(ohcX, motifsTensor2, strides = [1,4,1,1], padding='SAME', name="motifScan") 
motifScanRCTensor= tf.nn.conv2d(ohcX,motifsRCTensor2, strides = [1,4,1,1], padding='SAME', name="motifScanRC") 
##outputs [None,1,seqLen,numMotifs]

if args.interactions>0:
	#the following code implements TF cooperativity/competition
	#learned coeficients act as a scale of the log(kd) using nearby Kds for other (and the same) factors. For instance, for Kd_i_p of factor i at position p, log(Kdcoop_i_p) = log(Kd_i_p) + sum[all nearby Kds at distance d for factor j] ( C_i_j_d * Kd_j_(p+d) )
	#these are the kd cooperativity filters
	#here, first F/R refers to current motif [dim 3], second F/R refers to interacting motif [0];
	#for these parameters, Ccoop<0 implies competition, Ccoop>0 implies cooperation
	kdBonusInteractionsFilterFF = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsFF") 
	kdBonusInteractionsFilterFR = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsFR") 
	kdBonusInteractionsFilterRF = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsRF") 
	kdBonusInteractionsFilterRR = tf.Variable(tf.zeros([args.numMotifs, args.interactions,1,args.numMotifs]),name="tftfInteractionsRR") 
	#reshape motifScanTensor from [None, 1, seqLen, numMotifs] to [None, numMot, seqLen, 1]
	motifScanTensorReshape = tf.transpose(motifScanTensor, perm = [0,3,2,1]); # apply filter to these existing Kds # [None, numMot, seqLen, 1]
	motifScanRCTensorReshape = tf.transpose(motifScanRCTensor, perm = [0,3,2,1]);
	#apply filters
	#Each of these are [None, 1, seqLen, numMot]
	kdBonusScanTensorFF = tf.nn.conv2d(motifScanTensorReshape, kdBonusInteractionsFilterFF, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanFF") # interactions between current F motif [dim=3] and all other F motifs
	kdBonusScanTensorFR = tf.nn.conv2d(motifScanRCTensorReshape, kdBonusInteractionsFilterFR, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanFR") # interaction between current F motif and all other R motifs
	kdBonusScanTensorRF = tf.nn.conv2d(motifScanTensorReshape, kdBonusInteractionsFilterRF, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanRF") # interactions between current R motif [dim=3] and all other F motifs
	kdBonusScanTensorRR = tf.nn.conv2d(motifScanRCTensorReshape, kdBonusInteractionsFilterRR, strides = [1,args.numMotifs,1,1], padding='SAME', name="motifBonusScanRR") # interaction between current R motif and all other R motifs
	#add the bonus Kds from neighboring motifs to original motifs
	motifScanTensorWithBonus = tf.add(tf.add(kdBonusScanTensorFF, kdBonusScanTensorFR), motifScanTensor); #log(Kdcoop) = log(Kd) + log(Kdbonus) [None, 1, seqLen, numMot]
	motifScanRCTensorWithBonus = tf.add(tf.add(kdBonusScanTensorRF, kdBonusScanTensorRR), motifScanRCTensor); 
	logKdConcRatioTensor = tf.sub(logConcs,motifScanTensorWithBonus) # [None, 1, seqLen,numMotifs] 
	logKdConcRatioRCTensor = tf.sub(logConcs,motifScanRCTensorWithBonus) # [None, 1, seqLen,numMotifs] 
else:
	logKdConcRatioTensor = tf.sub(logConcs,motifScanTensor) # [None, 1, seqLen,numMotifs] 
	logKdConcRatioRCTensor = tf.sub(logConcs,motifScanRCTensor) # [None, 1, seqLen,numMotifs] 


pNotBoundTensor = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioTensor))); # size: [None,1,seqLen,numMotifs]
pNotBoundRCTensor = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioRCTensor))); # size: [None,1,seqLen,numMotifs]

if args.useEBound>0: 
	if args.verbose>0: sys.stderr.write("Using E-bound\n")
	epBoundTensor = tf.add(tf.reduce_sum(tf.sub(1.0,pNotBoundRCTensor), reduction_indices=[1,2]),tf.reduce_sum(tf.sub(1.0,pNotBoundTensor), reduction_indices=[1,2])) # size: [None, numMotifs] #expected amount of binding
else:
	if args.verbose>0: sys.stderr.write("Using P-bound\n")
	epBoundTensor = tf.sub(1.0,tf.mul(tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2]),tf.reduce_prod(pNotBoundTensor, reduction_indices=[1,2]))) # size: [None, numMotifs] # p(bound)

## POTENTIATION
if args.potentiation>0:
	if args.verbose>0: sys.stderr.write("Using potentiation layer\n")
	initPotent = np.random.standard_normal(args.numMotifs).astype(np.float32);
	#input default concentrations if applicable
	if args.initPotent is not None:
		vals = np.loadtxt(args.initPotent, dtype = np.str);
		initPotent[0:vals.shape[0]] = vals[:,1].astype("float32");
	if args.noTrainPotentiations==0:
		if args.verbose>0: sys.stderr.write("Training potentiations\n")
		potentiation = tf.Variable(tf.convert_to_tensor(initPotent.reshape([args.numMotifs])),name="potents");
	else:
		if args.verbose>0: sys.stderr.write("Not training potentiations\n")
		potentiation = tf.constant(initPotent.reshape([args.numMotifs]),name="potents");
	seqPotentialByTFTensor = tf.mul(epBoundTensor, potentiation); #size: [None,numMotifs]
	constantPot = tf.Variable(tf.zeros(1),name="constantPot")
	seqPotentialTensor = tf.sigmoid(tf.add(tf.reduce_sum(seqPotentialByTFTensor,reduction_indices=[1]), constantPot)) #[None, 1]
else:
	if args.verbose>0: sys.stderr.write("Not using potentiation layer\n")

if args.trainPositionalActivities>0: # account for positional activity with linear scaling of activity
	if args.trainStrandedActivities>0: # account for strand-specific activity biases
		pBoundPerPos = tf.sub(1.0,pNotBoundTensor) # size: [None,1,seqLen,numMotifs]
		pBoundPerPosRC = tf.sub(1.0,pNotBoundRCTensor) # size: [None,1,seqLen,numMotifs]
		if args.potentiation>0:
			pBoundPerPos = tf.transpose(tf.mul(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
			pBoundPerPosRC = tf.transpose(tf.mul(tf.transpose(pBoundPerPosRC, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
		#print(tf.Tensor.get_shape(pBoundPerPos))
		#print(tf.Tensor.get_shape(positionalActivity))
		expectedActivity = tf.matmul(tf.reshape(pBoundPerPos, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivity, (args.seqLen*args.numMotifs,1))) # size: [None,1]
		expectedActivityRC = tf.matmul(tf.reshape(pBoundPerPosRC, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivityRC, (args.seqLen*args.numMotifs,1))) # size: [None,1]
		expectedActivity = tf.add(expectedActivity, expectedActivityRC);
	else:
		if args.useEBound>0:
			pBoundPerPos = tf.add(tf.sub(1.0,pNotBoundTensor), tf.sub(1.0,pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
		else:
			pBoundPerPos = tf.sub(1.0,tf.mul(pNotBoundTensor, pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
		#print(tf.Tensor.get_shape(pBoundPerPos))
		if args.potentiation>0:
			pBoundPerPos = tf.transpose(tf.mul(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: [None,1,seqLen,numMotifs]
		if args.bindingLimits>0:
			pBoundPerPos = tf.minimum(pBoundPerPos, bindingLimits); #[None,1,seqLen,numMotifs]
		#print(tf.Tensor.get_shape(pBoundPerPos))
		#print(tf.Tensor.get_shape(positionalActivity))
		expectedActivity = tf.matmul(tf.reshape(pBoundPerPos, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivity, (args.seqLen*args.numMotifs,1))) # size: [None,1]
else: #no positional activities
	if args.trainStrandedActivities>0: # account for strand-specific activity biases
		if args.useEBound>0: 
			epBoundTensor = tf.reduce_sum(tf.sub(1.0,pNotBoundTensor), reduction_indices=[1,2]) # size: [None, numMotifs] #expected amount of binding
			epBoundTensorRC = tf.reduce_sum(tf.sub(1.0,pNotBoundRCTensor), reduction_indices=[1,2]) # size: [None, numMotifs] #expected amount of binding
		else:
			epBoundTensor = tf.sub(1.0,tf.reduce_prod(pNotBoundTensor,reduction_indices=[1,2])) # size: [None, numMotifs] # p(bound)
			epBoundTensorRC = tf.sub(1.0,tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2])) # size: [None, numMotifs] # p(bound)
		if args.potentiation>0:
			epBoundTensor = tf.transpose(tf.mul(tf.transpose(epBoundTensor), seqPotentialTensor)); # [None, numMotifs]
			epBoundTensorRC = tf.transpose(tf.mul(tf.transpose(epBoundTensorRC), seqPotentialTensor)); # [None, numMotifs]
			#print(tf.Tensor.get_shape(epBoundTensor))
			#print(tf.Tensor.get_shape(epBoundTensorRC))
		if args.bindingLimits>0:
			epBoundTensor = tf.minimum(epBoundTensor, bindingLimits); #[None, numMotifs]
		expectedActivity = tf.add(tf.matmul(epBoundTensor, tf.reshape(activities,(args.numMotifs,1))), tf.matmul(epBoundTensorRC, tf.reshape(activitiesRC,(args.numMotifs,1)))) #[None,1]
	else: #no positional or strand effects
		if args.potentiation>0:
			epBoundTensor = tf.transpose(tf.mul(tf.transpose(epBoundTensor),seqPotentialTensor)); # [None,numMotifs]
		if args.bindingLimits>0:
			epBoundTensor = tf.minimum(epBoundTensor, bindingLimits); #[None, numMotifs]
		expectedActivity = tf.matmul(epBoundTensor, tf.reshape(activities,(args.numMotifs,1))); #size: [None,1]

constant = tf.Variable(tf.zeros(1),name="constant")

predELY= tf.add(tf.reshape(expectedActivity, [-1]),constant) #size: [None]
realELY = tf.placeholder(tf.float32, [None]);

EPSILON=0.0001

mseTF = tf.reduce_mean(tf.square(realELY - predELY))
if args.meanSDFile is not None:
	vals = np.loadtxt(args.meanSDFile, dtype = np.str);
	llMeans = vals[:,0].astype("float32");
	llSDs = vals[:,1].astype("float32");
	maxllMean = llMeans.max()
	llSDLen = llSDs.shape[0];
	if args.verbose>0: sys.stderr.write("Minimizing the negative log liklihood with an SD vector of length %i and a maximum value of %f\n"%(llSDLen,maxllMean))
	llSDTensor = tf.constant(llSDs, name="llSDs")
	predELLLIndeces = tf.minimum(llSDLen-1, tf.maximum(0,tf.to_int32(tf.round(tf.mul(predELY,(llSDLen/maxllMean))))), name="predELLLInd") 
	predELYSDs = tf.nn.embedding_lookup(llSDTensor, predELLLIndeces, name="predELYSDs") #[None]
	predZ = tf.div(tf.sub(predELY, realELY), predELYSDs, name="predZ")
	negLogLik = tf.reduce_sum(tf.square(predZ), name="negLogLik")
	myLoss = negLogLik;
else:
	myLoss = mseTF;
if args.L2 is not None and args.noTrainMotifs==0:
	if args.verbose>0: sys.stderr.write("Using L2 regularization of PWMs with lambda=%s\n"%(args.L2))
	args.L2 = float(args.L2);
	paramPenaltyL2Tensor = tf.nn.l2_loss(motifsTensor) #doesn't make sense to l2 loss the concentrations since this brings them to 0; however, bringing the PWM entries to 0 does make sense.
	myLoss = tf.add(myLoss, tf.mul(paramPenaltyL2Tensor,args.L2));

if args.L1 is not None:
	if args.verbose>0: sys.stderr.write("Using L1 regularization of activities with lambda=%s\n"%(args.L1))
	args.L1 = float(args.L1);
	paramPenaltyL1Tensor = tf.reduce_sum(tf.abs(activities))
	paramNumActivityTensor = tf.reduce_sum(tf.cast(tf.greater(tf.abs(activities),EPSILON),tf.int32))
	if args.potentiation>0:
		paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(potentiation))
		paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(potentiation),EPSILON),tf.int32))
	if args.trainStrandedActivities>0:
		paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(activityDiffs))
		paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(activityDiffs),EPSILON),tf.int32))
	if args.trainPositionalActivities>0:
		paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(activitySlopes))
		paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(activitySlopes),EPSILON),tf.int32))
		if args.trainStrandedActivities>0:
			paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(activitySlopesDiff))
			paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(activitySlopesDiff),EPSILON),tf.int32))
	myLoss = tf.add(myLoss, tf.mul(paramPenaltyL1Tensor,args.L1));

if args.interactions>0:
	if args.verbose>0: sys.stderr.write("Using L1 regularization of interaction coeficients with lambda=%s\n"%(args.L1int))
 	paramPenaltyL1intTensor = tf.reduce_sum(tf.abs(kdBonusInteractionsFilterFF)) + tf.reduce_sum(tf.abs(kdBonusInteractionsFilterFR)) +tf.reduce_sum(tf.abs(kdBonusInteractionsFilterRF)) +tf.reduce_sum(tf.abs(kdBonusInteractionsFilterRR));
 	paramNumIntTensor = tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterFF),EPSILON),tf.int32))+tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterFR),EPSILON),tf.int32))+tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterRF),EPSILON),tf.int32))+tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterRR),EPSILON),tf.int32))
	myLoss = tf.add(myLoss, tf.mul(paramPenaltyL1intTensor,args.L1int));

opt = tf.train.AdamOptimizer(args.learningRate);
train_step = opt.minimize(myLoss);

#raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads));
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()
sess.run(init_op)
if args.restore is not None:
	if args.verbose>1:
		sys.stderr.write("Loading initial parameter settings: %s\n"%(args.restore))
	reader = tf.train.NewCheckpointReader(args.restore);
	if args.verbose>1:
		sys.stderr.write("Loading these variables: %s\n"%(", ".join([k.name for k in tf.all_variables() if k.name in ["%s:0"%s for s in reader.get_variable_to_shape_map().keys()]])))
	restorer = tf.train.Saver([k for k in tf.all_variables() if k.name in ["%s:0"%s for s in reader.get_variable_to_shape_map().keys()]]);
	restorer.restore(sess, args.restore)

def saveParams(sess):
	global args;
	global motifsTensor2;
	global logConcs;
	global activities;
	global constant;
	if args.noTrainMotifs==0:
		if (args.outFPre is None):
			outFile= sys.stdout;
		else:
			outFile = MYUTILS.smartGZOpen(args.outFPre+".pkdms",'w');
		#print out weight matrices
		pkdms = motifsTensor2.eval(session=sess); #[4,args.motifLen,1,args.numMotifs]);
		for i in range(0, args.numMotifs):
			outFile.write("#Motif %i\n"%(i));
			for j in range(0,4):
				outFile.write("#%s"%BASES[j]);
				for k in range(0,args.motifLen):
					outFile.write("\t%g"%(pkdms[j,k,0,i]));
				outFile.write("\n");
		outFile.write("\n");
		outFile.close();
	if args.interactions>0:
		global kdBonusInteractionsFilterFF, kdBonusInteractionsFilterFR, kdBonusInteractionsFilterRF, kdBonusInteractionsFilterRR;
		if (args.outFPre is None):
			outFile= sys.stdout;
		else:
			outFile = MYUTILS.smartGZOpen(args.outFPre+".interactions",'w');
		outFile.write("TF1\tTF2\tstrands\tdist\tcoef\n")
		strands = ["++","+-","-+","--"];
		interactionTensors = [kdBonusInteractionsFilterFF, kdBonusInteractionsFilterFR, kdBonusInteractionsFilterRF, kdBonusInteractionsFilterRR];
		for s in range(0,4):
			strand =strands[s]
			tfInteractions = interactionTensors[s].eval(session=sess); #[numMot, distance, 1, numMot]
			for i in range(0, args.numMotifs): # primary motif
				for j in range(0, args.numMotifs): # interacting motif
					for d in range(0, args.interactions):
						if np.abs(tfInteractions[j,d,0,i])>EPSILON:
							outFile.write("%i\t%i\t%s\t%i\t%g\n"%(i,j,strand,d,tfInteractions[j,d,0,i]));
		outFile.close();
	if (args.outFPre is None):
		outFile= sys.stdout;
	else:
		outFile = MYUTILS.smartGZOpen(args.outFPre+".params",'w');
	#print params
	concs = logConcs.eval(session=sess).reshape((args.numMotifs));
	activityVals = activities.eval(session=sess).reshape((args.numMotifs));
	outFile.write("i\tlogConc\tactivity");
	if args.potentiation>0:
		global potentiation;
		global constantPot;
		outFile.write("\tpotentiations");
		potentiationVals = potentiation.eval(session=sess).reshape((args.numMotifs));
	if args.bindingLimits>0:
		global bindingLimits
		outFile.write("\tbindingLimits");
		bindingLimitVals = bindingLimits.eval(session=sess).reshape((args.numMotifs));
	if args.trainStrandedActivities>0:
		global activityDiffs;
		outFile.write("\tactivityDiffs");
		activityDiffVals = activityDiffs.eval(session=sess).reshape((args.numMotifs));
	if args.trainPositionalActivities>0:
		global activitySlopes;
		outFile.write("\tactivitySlopes");
		activitySlopeVals = activitySlopes.eval(session=sess).reshape((args.numMotifs));
		if args.trainStrandedActivities>0:
			global activitySlopesDiff;
			outFile.write("\tactivitySlopeDiffs");
			activitySlopeDiffVals = activitySlopesDiff.eval(session=sess).reshape((args.numMotifs));
	outFile.write("\n");
	outFile.write("-1\tNA\t%g"%constant.eval(session=sess));#intercepts
	if args.potentiation>0:
		outFile.write("\t%g"%(constantPot.eval(session=sess)));
	if args.trainStrandedActivities>0:
		outFile.write("\tNA");
	if args.trainPositionalActivities>0:
		outFile.write("\tNA");
		if args.trainStrandedActivities>0:
			outFile.write("\tNA");
	outFile.write("\n");
	for i in range(0,args.numMotifs):
		outFile.write("%i\t%g\t%g"%(i, concs[i], activityVals[i]));#intercepts
		if args.potentiation>0:
			outFile.write("\t%g"%(potentiationVals[i]));
		if args.bindingLimits>0:
			outFile.write("\t%g"%(bindingLimitVals[i]));
		if args.trainStrandedActivities>0:
			outFile.write("\t%g"%(activityDiffVals[i]));
		if args.trainPositionalActivities>0:
			outFile.write("\t%g"%(activitySlopeVals[i]));
			if args.trainStrandedActivities>0:
				outFile.write("\t%g"%(activitySlopeDiffVals[i]));
		outFile.write("\n");
	outFile.close();

