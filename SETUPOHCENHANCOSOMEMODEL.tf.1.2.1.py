
if args.accIsAct>0 and args.potentiation==0:
	raise Exception("Cannot include accessibility influence on EL without including potentiation layer!")

if not hasattr(args, 'L2Pos'):
	args.L2Pos=0.001;
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
	positionalActivityBias = tf.Variable(tf.ones([args.seqLen,args.numMotifs]), name="positionalActivities") #[seqLen, numMotifs]
	positionalActivity = tf.multiply(positionalActivityBias, activities) #[seqLen, numMotifs]
	if args.trainStrandedActivities:
		positionalActivityBiasRC = tf.Variable(tf.ones([args.seqLen,args.numMotifs]), name="positionalActivitiesRC") #[seqLen, numMotifs]
		positionalActivityRC = tf.multiply(positionalActivityBiasRC, activitiesRC) #[seqLen, numMotifs]

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

logKdConcRatioTensor = tf.subtract(logConcs,motifScanTensor) # [None, 1, seqLen,numMotifs] 
logKdConcRatioRCTensor = tf.subtract(logConcs,motifScanRCTensor) # [None, 1, seqLen,numMotifs] 
pNotBoundTensor1 = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioTensor))); # size: [None,1,seqLen,numMotifs]
pNotBoundRCTensor1 = tf.div(1.0,tf.add(1.0,tf.exp(logKdConcRatioRCTensor))); # size: [None,1,seqLen,numMotifs]

if args.interactions>0:
	#the following code implements TF cooperativity/competition; see lab notebook "How best to model TF-TF interactions"
	#for cooperativity, a Kd scale is added in proportion to the bidning of the cofactor (log scaled)
	#for competition, a negative Kd scale is added that scales with the Kd of the competing factor (standard space)
	#negative coeficients indicate competition between the factors at the specified distance
	#positive coeficients indicate cooperativity
	#pBoundX = 1-(1/(1+exp(logXs-CxyComp*PBY*(logY))*(1+(exp(CxyCoop)-1)*PBY)))
	#these are the kd interaction filters
	#here, first F/R refers to current motif X [dim 3], second F/R refers to interacting motif Y [0];
	#for these parameters, Ccoop<0 implies competition, Ccoop>0 implies cooperation
	#kdBonusInteractionsFilterFF = tf.Variable(tf.random_normal([args.numMotifs, args.interactions,1,args.numMotifs],stddev=0.01),name="tftfInteractionsFF") #[numMotifs,interactions,1,numMotifs] = [1,X,interactions,1,Y]
	#kdBonusInteractionsFilterFR = tf.Variable(tf.random_normal([args.numMotifs, args.interactions,1,args.numMotifs],stddev=0.01),name="tftfInteractionsFR") 
	#kdBonusInteractionsFilterRF = tf.Variable(tf.random_normal([args.numMotifs, args.interactions,1,args.numMotifs],stddev=0.01),name="tftfInteractionsRF") 
	#kdBonusInteractionsFilterRR = tf.Variable(tf.random_normal([args.numMotifs, args.interactions,1,args.numMotifs],stddev=0.01),name="tftfInteractionsRR") 
	kdBonusInteractionsFilterFF = tf.Variable(tf.fill([args.numMotifs, args.interactions,1,args.numMotifs],-0.001),name="tftfInteractionsFF") #[numMotifs,interactions,1,numMotifs] = [1,X,interactions,1,Y]
	kdBonusInteractionsFilterFR = tf.Variable(tf.fill([args.numMotifs, args.interactions,1,args.numMotifs],-0.001),name="tftfInteractionsFR") 
	kdBonusInteractionsFilterRF = tf.Variable(tf.fill([args.numMotifs, args.interactions,1,args.numMotifs],-0.001),name="tftfInteractionsRF") 
	kdBonusInteractionsFilterRR = tf.Variable(tf.fill([args.numMotifs, args.interactions,1,args.numMotifs],-0.001),name="tftfInteractionsRR") 
	#weights for cooperativity
	coopWeightsFilterFF = tf.subtract(tf.exp(tf.nn.relu(kdBonusInteractionsFilterFF)),1, name="coopParamsFF")
	coopWeightsFilterFR = tf.subtract(tf.exp(tf.nn.relu(kdBonusInteractionsFilterFR)),1, name="coopParamsFR")
	coopWeightsFilterRF = tf.subtract(tf.exp(tf.nn.relu(kdBonusInteractionsFilterRF)),1, name="coopParamsRF")
	coopWeightsFilterRR = tf.subtract(tf.exp(tf.nn.relu(kdBonusInteractionsFilterRR)),1, name="coopParamsRR")
	#weights for competition
	compWeightsFilterFF = tf.nn.relu(tf.negative(kdBonusInteractionsFilterFF), name="compParamsFF") # all positive, so I will subtract these later
	compWeightsFilterFR = tf.nn.relu(tf.negative(kdBonusInteractionsFilterFR), name="compParamsFR") # all positive, so I will subtract these later
	compWeightsFilterRF = tf.nn.relu(tf.negative(kdBonusInteractionsFilterRF), name="compParamsRF") # all positive, so I will subtract these later
	compWeightsFilterRR = tf.nn.relu(tf.negative(kdBonusInteractionsFilterRR), name="compParamsRR") # all positive, so I will subtract these later
	#pBoundX = 1-(1/(1+exp(logXs-compWF[X,I,1,Y]*PBY[,1,L,Y]*(logY[,1,L,Y]))*(1+(coopWF[X,I,1,Y]*PBY[,1,L,Y]))))
	#pBounds
	pBoundTensor1 = tf.subtract(1.0, pNotBoundTensor1)#[None,1,seqLen,numMotifs]
	pBoundRCTensor1 = tf.subtract(1.0, pNotBoundRCTensor1) #[None,1,seqLen,numMotifs] 
	#logY*PBY:
	logKDConcPBYTensor = tf.transpose(tf.multiply(pBoundTensor1, logKdConcRatioTensor), perm = [0,3,2,1])#[None,1,seqLen,numMotifs] -> [None, numMot, seqLen, 1]
	logKDConcPBYRCTensor = tf.transpose(tf.multiply(pBoundRCTensor1, logKdConcRatioRCTensor), perm = [0,3,2,1])#[None,1,seqLen,numMotifs] -> [None, numMot, seqLen, 1]
	pBoundTensor1 = tf.transpose(pBoundTensor1, perm = [0,3,2,1])     #[None, numMot, seqLen, 1]
	pBoundRCTensor1 = tf.transpose(pBoundRCTensor1, perm = [0,3,2,1]) #[None, numMot, seqLen, 1]
	#pBoundX = 1-(1/(1+exp(logXs-compWF[Y,I,1,X]*logYPBY[,Y,L,1]))*(1+(coopWF[Y,I,1,X]]*PBY[,Y,L,1]))))
	#the following contain the weighted cooperative and competitive effects for all potential binding sites and at all interaction distances between X and Y
	#apply convolution here to make these for i in seqLen refer to X positions
	#compEffects = compWF[Y,I,1,X]*logYPBY[,Y,L,1]
	compEffectsFFTensor = tf.nn.conv2d(logKDConcPBYTensor,   compWeightsFilterFF, strides = [1,args.numMotifs,1,1], padding='SAME', name="compEffectsFF")  #[None,1,seqLen,X]
	compEffectsRFTensor = tf.nn.conv2d(logKDConcPBYTensor,   compWeightsFilterRF, strides = [1,args.numMotifs,1,1], padding='SAME', name="compEffectsRF")  #[None,1,seqLen,X]
	compEffectsFRTensor = tf.nn.conv2d(logKDConcPBYRCTensor, compWeightsFilterFR, strides = [1,args.numMotifs,1,1], padding='SAME', name="compEffectsFR")  #[None,1,seqLen,X]
	compEffectsRRTensor = tf.nn.conv2d(logKDConcPBYRCTensor, compWeightsFilterRR, strides = [1,args.numMotifs,1,1], padding='SAME', name="compEffectsRR")  #[None,1,seqLen,X]
	#coopEffects = coopWF[Y,I,1,X]]*PBY[,Y,L,1]
	coopEffectsFFTensor = tf.nn.conv2d(pBoundTensor1,   coopWeightsFilterFF, strides = [1,args.numMotifs,1,1], padding='SAME', name="coopEffectsFF")  #[None,1,seqLen,X]
	coopEffectsRFTensor = tf.nn.conv2d(pBoundTensor1,   coopWeightsFilterRF, strides = [1,args.numMotifs,1,1], padding='SAME', name="coopEffectsRF")  #[None,1,seqLen,X]
	coopEffectsFRTensor = tf.nn.conv2d(pBoundRCTensor1, coopWeightsFilterFR, strides = [1,args.numMotifs,1,1], padding='SAME', name="coopEffectsFR")  #[None,1,seqLen,X]
	coopEffectsRRTensor = tf.nn.conv2d(pBoundRCTensor1, coopWeightsFilterRR, strides = [1,args.numMotifs,1,1], padding='SAME', name="coopEffectsRR")  #[None,1,seqLen,X]
	#pBoundX = 1-(1/(1+exp(logXs-compEffects)*(1+coopEffects)))
	#these now contain the cooperative and competititve effects for each TF in each position - use the logX to calculate PBound_X
	#add the bonus Kds from neighboring motifs to original motifs
	pNotBoundTensor = tf.div(1.0,tf.add(1.0,tf.multiply(tf.exp(tf.subtract(logKdConcRatioTensor, tf.add(compEffectsFFTensor,compEffectsFRTensor))),tf.add(1.0,tf.add(coopEffectsFFTensor,coopEffectsFRTensor)))))
	pNotBoundRCTensor = tf.div(1.0,tf.add(1.0,tf.multiply(tf.exp(tf.subtract(logKdConcRatioRCTensor, tf.add(compEffectsRFTensor,compEffectsRRTensor))),tf.add(1.0,tf.add(coopEffectsRFTensor,coopEffectsRRTensor)))))
else:
	pNotBoundTensor = pNotBoundTensor1;
	pNotBoundRCTensor = pNotBoundRCTensor1


if args.useEBound>0: 
	if args.verbose>0: sys.stderr.write("Using E-bound\n")
	epBoundTensor = tf.add(tf.reduce_sum(tf.subtract(1.0,pNotBoundRCTensor), reduction_indices=[1,2]),tf.reduce_sum(tf.subtract(1.0,pNotBoundTensor), reduction_indices=[1,2])) # size: [None, numMotifs] #expected amount of binding
else:
	if args.verbose>0: sys.stderr.write("Using P-bound\n")
	epBoundTensor = tf.subtract(1.0,tf.multiply(tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2]),tf.reduce_prod(pNotBoundTensor, reduction_indices=[1,2]))) # size: [None, numMotifs] # p(bound)

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
	seqPotentialByTFTensor = tf.multiply(epBoundTensor, potentiation); #size: [None,numMotifs]
	constantPot = tf.Variable(tf.zeros(1),name="constantPot")
	seqPotentialTensor = tf.sigmoid(tf.add(tf.reduce_sum(seqPotentialByTFTensor,reduction_indices=[1]), constantPot)) #[None, 1]
else:
	if args.verbose>0: sys.stderr.write("Not using potentiation layer\n")

if args.trainPositionalActivities>0: # account for positional activity with linear scaling of activity
	if args.trainStrandedActivities>0: # account for strand-specific activity biases
		pBoundPerPos = tf.subtract(1.0,pNotBoundTensor) # size: [None,1,seqLen,numMotifs]
		pBoundPerPosRC = tf.subtract(1.0,pNotBoundRCTensor) # size: [None,1,seqLen,numMotifs]
		if args.potentiation>0:
			pBoundPerPos = tf.transpose(tf.multiply(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
			pBoundPerPosRC = tf.transpose(tf.multiply(tf.transpose(pBoundPerPosRC, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: None,1,seqLen,numMotifs]
		#print(tf.Tensor.get_shape(pBoundPerPos))
		#print(tf.Tensor.get_shape(positionalActivity))
		if args.bindingLimits>0:
			expectedActivity = tf.reduce_sum(tf.multiply(pBoundPerPos, positionalActivity),reduction_indices=[1,2]) # size: [None,numMotifs]
			expectedActivityRC = tf.reduce_sum(tf.multiply(pBoundPerPosRC, positionalActivityRC),reduction_indices=[1,2]) # size: [None,numMotifs]
			expectedActivity = tf.reduce_sum(tf.add( #min of positive activities and max of negative activities accounting for binding limits.
				tf.nn.relu(                        tf.minimum(tf.reshape(tf.multiply(bindingLimits,activities),(1,args.numMotifs)),tf.add(expectedActivity, expectedActivityRC))), #positive
				tf.negative(tf.nn.relu(tf.negative(tf.maximum(tf.reshape(tf.multiply(bindingLimits,activities),(1,args.numMotifs)),tf.add(expectedActivity, expectedActivityRC))))) #negative
				),reduction_indices=[1]) #[None,1]
			
		else:
			expectedActivity = tf.matmul(tf.reshape(pBoundPerPos, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivity, (args.seqLen*args.numMotifs,1))) # size: [None,1]
			expectedActivityRC = tf.matmul(tf.reshape(pBoundPerPosRC, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivityRC, (args.seqLen*args.numMotifs,1))) # size: [None,1]
			expectedActivity = tf.add(expectedActivity, expectedActivityRC);
	else:
		if args.useEBound>0:
			pBoundPerPos = tf.add(tf.subtract(1.0,pNotBoundTensor), tf.subtract(1.0,pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
		else:
			pBoundPerPos = tf.subtract(1.0,tf.multiply(pNotBoundTensor, pNotBoundRCTensor)) # size: [None,1,seqLen,numMotifs]
		#print(tf.Tensor.get_shape(pBoundPerPos))
		if args.potentiation>0:
			pBoundPerPos = tf.transpose(tf.multiply(tf.transpose(pBoundPerPos, perm=(1,2,3,0)), seqPotentialTensor), perm = (3,0,1,2)) # size: [None,1,seqLen,numMotifs]
		if args.bindingLimits>0:
			pBoundPerPos = tf.minimum(pBoundPerPos, bindingLimits); #[None,1,seqLen,numMotifs]
		#print(tf.Tensor.get_shape(pBoundPerPos))
		#print(tf.Tensor.get_shape(positionalActivity))
		if args.bindingLimits>0:
			expectedActivity = tf.reduce_sum(tf.multiply(pBoundPerPos, positionalActivity),reduction_indices=[1,2]) # size: [None,numMotifs]
			expectedActivity = tf.reduce_sum(tf.add( #min of positive activities and max of negative activities accounting for binding limits.
				tf.nn.relu(                        tf.minimum(tf.reshape(tf.multiply(bindingLimits,activities),(1,args.numMotifs)),expectedActivity)), #positive
				tf.negative(tf.nn.relu(tf.negative(tf.maximum(tf.reshape(tf.multiply(bindingLimits,activities),(1,args.numMotifs)),expectedActivity)))) #negative
				),reduction_indices=[1]) #[None,1]
		else:
			expectedActivity = tf.matmul(tf.reshape(pBoundPerPos, (-1, args.seqLen*args.numMotifs)), tf.reshape(positionalActivity, (args.seqLen*args.numMotifs,1))) # size: [None,1]
else: #no positional activities
	if args.trainStrandedActivities>0: # account for strand-specific activity biases
		if args.useEBound>0: 
			epBoundTensor   = tf.reduce_sum(tf.subtract(1.0,  pNotBoundTensor), reduction_indices=[1,2]) # size: [None, numMotifs] #expected amount of binding
			epBoundTensorRC = tf.reduce_sum(tf.subtract(1.0,pNotBoundRCTensor), reduction_indices=[1,2]) # size: [None, numMotifs] #expected amount of binding
		else:
			epBoundTensor = tf.subtract(1.0,tf.reduce_prod(pNotBoundTensor,reduction_indices=[1,2])) # size: [None, numMotifs] # p(bound)
			epBoundTensorRC = tf.subtract(1.0,tf.reduce_prod(pNotBoundRCTensor,reduction_indices=[1,2])) # size: [None, numMotifs] # p(bound)
		if args.potentiation>0:
			epBoundTensor = tf.transpose(tf.multiply(tf.transpose(epBoundTensor), seqPotentialTensor)); # [None, numMotifs]
			epBoundTensorRC = tf.transpose(tf.multiply(tf.transpose(epBoundTensorRC), seqPotentialTensor)); # [None, numMotifs]
			#print(tf.Tensor.get_shape(epBoundTensor))
			#print(tf.Tensor.get_shape(epBoundTensorRC))
		if args.bindingLimits>0:#note that when adding strand-specific activities when there are binding limits, the output will change without changing the params because you can't limit both simultaneously now.
			expectedActivity = tf.reduce_sum(tf.add( #min of positive activities and max of negative activities accounting for binding limits.
				tf.nn.relu(                        tf.minimum(tf.reshape(tf.multiply(bindingLimits,activities),(1,args.numMotifs)),tf.add(tf.multiply(epBoundTensor, activities), tf.multiply(epBoundTensorRC, activitiesRC)))), #positive
				tf.negative(tf.nn.relu(tf.negative(tf.maximum(tf.reshape(tf.multiply(bindingLimits,activities),(1,args.numMotifs)),tf.add(tf.multiply(epBoundTensor, activities), tf.multiply(epBoundTensorRC, activitiesRC)))))) #negative
				),reduction_indices=[1]) #[None,1]
			#sys.stderr.write(" ".join([str(x) for x in tf.Tensor.get_shape(expectedActivity)])+"\n")
		else:
			expectedActivity = tf.add(tf.matmul(epBoundTensor, tf.reshape(activities,(args.numMotifs,1))), tf.matmul(epBoundTensorRC, tf.reshape(activitiesRC,(args.numMotifs,1)))) #[None,1]
	else: #no positional or strand effects
		if args.potentiation>0:
			epBoundTensor = tf.transpose(tf.multiply(tf.transpose(epBoundTensor),seqPotentialTensor)); # [None,numMotifs]
		if args.bindingLimits>0:
			epBoundTensor = tf.minimum(epBoundTensor, bindingLimits); #[None, numMotifs]
		expectedActivity = tf.matmul(epBoundTensor, tf.reshape(activities,(args.numMotifs,1))); #size: [None,1]

constant = tf.Variable(tf.zeros(1),name="constant")

if args.accIsAct>0:
	accActivity = tf.Variable(tf.zeros(1),name="accessActiv")
	accActivELTensor = tf.multiply(seqPotentialTensor, accActivity) # [None,1]
	predELY= tf.add(tf.add(tf.reshape(expectedActivity, [-1]),tf.reshape(accActivELTensor, [-1])), constant) #size: [None]
else:
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
	predELLLIndeces = tf.minimum(llSDLen-1, tf.maximum(0,tf.to_int32(tf.round(tf.multiply(predELY,(llSDLen/maxllMean))))), name="predELLLInd") 
	predELYSDs = tf.nn.embedding_lookup(llSDTensor, predELLLIndeces, name="predELYSDs") #[None]
	predZ = tf.div(tf.subtract(predELY, realELY), predELYSDs, name="predZ")
	negLogLik = tf.reduce_sum(tf.square(predZ), name="negLogLik")
	myLoss = negLogLik;
else:
	myLoss = mseTF;
if args.L2 is not None and args.noTrainMotifs==0:
	if args.verbose>0: sys.stderr.write("Using L2 regularization of PWMs with lambda=%s\n"%(args.L2))
	args.L2 = float(args.L2);
	paramPenaltyL2Tensor = tf.nn.l2_loss(motifsTensor) #doesn't make sense to l2 loss the concentrations since this brings them to 0; however, bringing the PWM entries to 0 does make sense.
	myLoss = tf.add(myLoss, tf.multiply(paramPenaltyL2Tensor,args.L2));

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
	if args.accIsAct>0:
		paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.abs(accActivity)
		paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(accActivity),EPSILON),tf.int32))
	if args.trainPositionalActivities>0:
		paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(tf.slice(positionalActivityBias,[0,0],[1,args.numMotifs]))) #penalize only the first column of positional activities
		#So only one position per TF is L1; the differences between the others are L2
		paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(tf.slice(positionalActivityBias,[0,0],[1,args.numMotifs])),EPSILON),tf.int32))
		if args.trainStrandedActivities>0:
			paramPenaltyL1Tensor = paramPenaltyL1Tensor + tf.reduce_sum(tf.abs(tf.slice(positionalActivityBiasRC,[0,0],[1,args.numMotifs]))) 
			paramNumActivityTensor = paramNumActivityTensor +tf.reduce_sum(tf.cast(tf.greater(tf.abs(tf.slice(positionalActivityBiasRC,[0,0],[1,args.numMotifs])),EPSILON),tf.int32))
	myLoss = tf.add(myLoss, tf.multiply(paramPenaltyL1Tensor,args.L1));

if args.L2Pos is not None and args.trainPositionalActivities>0:
	args.L2Pos = float(args.L2Pos);
	paramPenaltyL2PosTensor = tf.reduce_sum(tf.abs(tf.subtract(tf.slice(positionalActivityBias,[0,0],[args.seqLen-1,args.numMotifs]),tf.slice(positionalActivityBias,[1,0],[args.seqLen-1,args.numMotifs]))))
	if args.trainStrandedActivities>0:
		paramPenaltyL2PosTensor = paramPenaltyL2PosTensor + tf.nn.l2_loss(tf.subtract(tf.slice(positionalActivityBiasRC,[0,0],[args.seqLen-1,args.numMotifs]),tf.slice(positionalActivityBiasRC,[1,0],[args.seqLen-1,args.numMotifs])))
	myLoss = tf.add(myLoss, tf.multiply(paramPenaltyL2PosTensor, args.L2Pos));
	

if args.interactions>0:
	if args.verbose>0: sys.stderr.write("Using L1 regularization of interaction coeficients with lambda=%s\n"%(args.L1int))
 	paramPenaltyL1intTensor = tf.reduce_sum(tf.abs(kdBonusInteractionsFilterFF)) + tf.reduce_sum(tf.abs(kdBonusInteractionsFilterFR)) +tf.reduce_sum(tf.abs(kdBonusInteractionsFilterRF)) +tf.reduce_sum(tf.abs(kdBonusInteractionsFilterRR));
 	paramNumIntTensor = tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterFF),EPSILON),tf.int32))+tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterFR),EPSILON),tf.int32))+tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterRF),EPSILON),tf.int32))+tf.reduce_sum(tf.cast(tf.greater(tf.abs(kdBonusInteractionsFilterRR),EPSILON),tf.int32))
	myLoss = tf.add(myLoss, tf.multiply(paramPenaltyL1intTensor,args.L1int));

opt = tf.train.AdamOptimizer(args.learningRate);
train_step = opt.minimize(myLoss);

#raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads));
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init_op)
if args.restore is not None:
	if args.verbose>1:
		sys.stderr.write("Loading initial parameter settings: %s\n"%(args.restore))
	reader = tf.train.NewCheckpointReader(args.restore);
	if args.verbose>1:
		sys.stderr.write("Loading these variables: %s\n"%(", ".join([k.name for k in tf.global_variables() if k.name in ["%s:0"%s for s in reader.get_variable_to_shape_map().keys()]])))
	restorer = tf.train.Saver([k for k in tf.global_variables() if k.name in ["%s:0"%s for s in reader.get_variable_to_shape_map().keys()]]);
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
			outFile = MYUTILS.smartGZOpen(args.outFPre+".interactions.gz",'w');
		outFile.write("TF1\tTF2\tstrands\tdist\tcoef\n")
		strands = ["++","+-","-+","--"];
		interactionTensors = [kdBonusInteractionsFilterFF, kdBonusInteractionsFilterFR, kdBonusInteractionsFilterRF, kdBonusInteractionsFilterRR];
		for s in range(0,4):
			strand =strands[s]
			tfInteractions = interactionTensors[s].eval(session=sess); #[numMot, distance, 1, numMot]
			for i in range(0, args.numMotifs): # primary motif
				for j in range(0, args.numMotifs): # interacting motif
					for d in range(0, args.interactions):
						outFile.write("%i\t%i\t%s\t%i\t%g\n"%(i,j,strand,d-(args.interactions//2),tfInteractions[j,d ,0,i]));
		outFile.close();
	if args.trainPositionalActivities>0:
		if (args.outFPre is None):
			outFile= sys.stdout;
		else:
			outFile = MYUTILS.smartGZOpen(args.outFPre+".positional.gz",'w');
		global positionalActivityBias;
		outFile.write("TF\tposition\tpositionalActivityBias");
		positionalActivityBiasVals = positionalActivityBias.eval(session=sess).reshape((args.seqLen,args.numMotifs));
		if args.trainStrandedActivities>0:
			global positionalActivityBiasRC;
			outFile.write("\tpositionalActivityBiasRC");
			positionalActivityBiasRCVals = positionalActivityBiasRC.eval(session=sess).reshape((args.seqLen,args.numMotifs));
		outFile.write("\n");
		for j in range(0,args.numMotifs):
			for i in range(0,args.seqLen):
				outFile.write("%i\t%i\t%g"%(j,i,positionalActivityBiasVals[i,j]));
				if args.trainStrandedActivities>0:
					outFile.write("\t%g"%positionalActivityBiasRCVals[i,j]);
				outFile.write("\n");
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
	outFile.write("\n");
	if args.accIsAct>0:
		global accActivity #hide this as the concentration constant since this is unused otherwise
		outFile.write("-1\t%g\t%g"%(accActivity.eval(session=sess),constant.eval(session=sess)));#intercepts
	else:
		outFile.write("-1\tNA\t%g"%constant.eval(session=sess));#intercepts
	if args.potentiation>0:
		outFile.write("\t%g"%(constantPot.eval(session=sess)));
	if args.bindingLimits>0:
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
		outFile.write("\n");
	outFile.close();

