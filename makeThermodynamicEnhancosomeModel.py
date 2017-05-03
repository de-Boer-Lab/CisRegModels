#!/usr/bin/env python
import warnings
import sys
import argparse
parser = argparse.ArgumentParser(description='Creates a biochemical transcription model using OHC sequence input.')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file of one-hot-code sequences, preceeded by their expression value, tab delim', required=True);
parser.add_argument('-sl',dest='seqLen',	metavar='<seqLen>',help='Input sequence length in bp', required=True);
parser.add_argument('-b',dest='batch',	metavar='<batchSize>',help='Batch size to use to feed into the neural net [default = 1000] ', required=False, default = "1000");
parser.add_argument('-se',dest='saveEvery',	metavar='<saveEvery>',help='save neural net every X batches [default = 500] ', required=False, default = "500");
parser.add_argument('-res',dest='restore',	metavar='<restoreFromCkptFile>',help='Restore NN from file ', required=False);
parser.add_argument('-lpb',dest='learnsPerBatch',	metavar='<learnsPerBatch>',help='How many learning steps do we take at each batch [default = 1000] ', required=False, default = "1000");
parser.add_argument('-nm',dest='numMotifs',	metavar='<numMotifs>',help='The number of motifs to include in the model [default=400]', required=False, default = "400");
parser.add_argument('-ml',dest='motifLen',	metavar='<motifLen>',help='Motif length [default = 20]', required=False, default = "20");
parser.add_argument('-dm',dest='motifsFP',	metavar='<defaultMotifFile>',help='Pre-load the PWMs with these', required=False);
parser.add_argument('-mll',dest='meanSDFile',	metavar='<meanSDRelationshipFile>',help='If this is specified, instead of the MSE, minimize the negative log liklihood and use this curve to calculate predicted SDs for promoters.', required=False);
parser.add_argument('-ic',dest='initConcs', metavar='<initConcs>',help='Initialize log concentrations to these (usually min Kds for the PWM; same order as others) [default log(concs)=0]', required=False);
parser.add_argument('-ia',dest='initActiv', metavar='<initActiv>',help='Initialize activities to these [defaults to gaussian]', required=False);
parser.add_argument('-ip',dest='initPotent', metavar='<initPotent>',help='Initialize potentiations to these [defaults to gaussian]', required=False);
parser.add_argument('-eb',dest='useEBound', action='count',help='use expected binding rather than prob binding?', required=False, default=0);
parser.add_argument('-bl',dest='bindingLimits', action='count',help='Include a maximum binding parameter?', required=False, default=0);
parser.add_argument('-ibl',dest='initBindLim', metavar='<initBindLim>',help='Initialize binding limits to these [defaults to 1]', required=False);
parser.add_argument('-po',dest='potentiation', action='count',help='add a layer of potentiation, modulating accessibility?', required=False, default=0);
parser.add_argument('-tftf',dest='interactions', metavar='<windowSize>',help='Add a layer of interaction filters to scale Kds based on nearby Kds within this window [default = do not use interaction terms]', required=False, default=0);
parser.add_argument('-posa',dest='trainPositionalActivities',action='count',help='Train slopes capturing a linear positional dependence? [default=no]. Note that this necessarily makes the model an EBound model after the potentiation step', required=False, default=0);
parser.add_argument('-stra',dest='trainStrandedActivities',action='count',help='Train activities capturing strand-specific differences in activity? [default=no]', required=False, default=0);
parser.add_argument('-ntc',dest='noTrainConcs', action='count',help='make TF concentrations constant (i.e. whatever concentrations are input)?', required=False, default=0);
parser.add_argument('-nta',dest='noTrainActivities', action='count',help='make TF activities constant (i.e. whatever activities are input)?', required=False, default=0);
parser.add_argument('-ntp',dest='noTrainPotentiations', action='count',help='make TF potentiations constant (i.e. whatever potentiations are input)?', required=False, default=0);
parser.add_argument('-ntm',dest='noTrainMotifs', action='count',help='Don\'t change the motif layer of variables?', required=False, default=0);
parser.add_argument('-ntbl',dest='noTrainBL', action='count',help='Don\'t change the binding limits?', required=False, default=0);
parser.add_argument('-r',dest='runs',	metavar='<numRuns>',help='How many times to run through data [default=1]', required=False, default="1");
parser.add_argument('-lr',dest='learningRate',	metavar='<learningRate>',help='the learning rate parameter (bigger = learns faster, but noisier) [default=0.01]', required=False, default="0.01");
parser.add_argument('-l1',dest='L1',	metavar='<l1Penalty>',help='L1-regularization parameter for the activities and potentiations (good values are ~0.00001 [default=no regularization]', required=False);
parser.add_argument('-l1int',dest='L1int',	metavar='<l1PenaltyForInteractions>',help='L1-regularization parameter for the TF-TF interaction terms [default=0.00001]', required=False, default="0.00001");
parser.add_argument('-l2',dest='L2',	metavar='<l2Penalty>',help='L2-regularization parameter for the PWMs (good values are ~0.000001) [default=no regularization]', required=False);
parser.add_argument('-t',dest='threads',	metavar='<threads>',help='Number of threads to make use of [default=1]',default = "1", required=False);
parser.add_argument('-o',dest='outFPre', metavar='<outFilePrefix>',help='Where to output results - prefix [default=stdout]', required=False);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);
#sys.argv = "makeThermodynamicEnhancosomeModel.py -i 20160609_average_promoter_ELs_per_seq_pTpA_ALL.shuffled_OHC_test.txt.gz   -o test.out    -v -v -v -sl 110 -lpb 10 -b 1024 -l1 0.000001 -lr 0.01 -r 1 -t 1 -nm 300 -ml 22 -dm /home/unix/cgdeboer/CIS-BP/YeTFaSCo/allTF_PKdMFiles.txt -po -ia PBound_learned_activities_pTpA_ACS.txt -ic PBound_learned_concs_pTpA_ACS.txt -tftf 59 -posa -stra ".split();
##sys.argv = "makeThermodynamicEnhancosomeModel.py -i 20160609_average_promoter_ELs_per_seq_pTpA_ALL.shuffled_OHC_train.txt.gz             -o PBound_learned_params_pTpA_ACSPOLI.txt     -v -v -v  -sl 110 -lpb 10 -b 1024 -l1 0.00001 -l2 0.000001 -r 1 -t 16 -nm 244 -ml 25 -dm /home/unix/cgdeboer/CIS-BP/YeTFaSCo/allTF_PKdMFiles.txt  -po -ia PBound_learned_activities_pTpA_ACSP.txt     -ic PBound_learned_concs_pTpA_ACSP.txt     -ip PBound_learned_potentiations_pTpA_ACSP.txt -tftf 59 -stra -posa -ntm".split();
args = parser.parse_args();
import MYUTILS
import PWM;
import tensorflow as tf
import numpy as np;

args.batch = int(args.batch);
args.seqLen = int(args.seqLen);
args.numMotifs = int(args.numMotifs);
args.motifLen = int(args.motifLen);
args.learnsPerBatch = int(args.learnsPerBatch);
args.saveEvery = int(args.saveEvery);
args.learningRate = float(args.learningRate);
args.threads = int(args.threads);
args.interactions = int(args.interactions);
args.L1int = float(args.L1int);

if (args.logFP is not None):
	logFile=MYUTILS.smartGZOpen(args.logFP,'w');
	sys.stderr=logFile;

if (args.outFPre is not None):
	if args.verbose>0: sys.stderr.write("Outputting to file "+args.outFPre+"*\n");

with open('/home/unix/cgdeboer/lib/python/SETUPOHCENHANCOSOMEMODEL.py') as fd:
	exec(fd.read())

saveParams(sess)
trainingSteps=0;
b = 0
batchX = np.zeros((args.batch,4,args.seqLen,1))
batchY = np.zeros((args.batch))
for r in range(0, int(args.runs)):
	if args.verbose>1:
		sys.stderr.write("	Run %i...\n"%(r));
	inFile=MYUTILS.smartGZOpen(args.inFP,'r');
	for line in inFile:
		if line is None or line == "" or line[0]=="#": continue
		curData = np.fromstring(line, dtype=float, sep="\t")
		batchY[b]=curData[0];
		batchX[b,:,:,0] = curData[1:].reshape((4,args.seqLen))
		b+=1
		if b==args.batch:
			if args.verbose>2:
				sys.stderr.write("		Current batch %i; looked at %i examples...\n"%(trainingSteps, trainingSteps * args.batch));
			#train with current data
			for j in range(0,args.learnsPerBatch):
				sess.run(train_step, feed_dict={ohcX: batchX, realELY: batchY})
				if args.verbose>3: 
					if args.L1 is not None:
						sys.stderr.write("			cur train MSE = %f;\tloss = %f;\tmodel paramsum = %f;\tparamsnum = %i\n"%(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY}),myLoss.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY}),paramPenaltyL1Tensor.eval(session=sess), paramNumActivityTensor.eval(session=sess)));
					else:
						sys.stderr.write("			cur train MSE = %f\n"%(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY})));
			if args.verbose>2: 
				if args.L1 is not None:
					sys.stderr.write("		cur batch MSE = %f;\tloss = %f;\tmodel paramsum = %f;\tparamsnum = %i\n"%(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY}),myLoss.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY}),paramPenaltyL1Tensor.eval(session=sess), paramNumActivityTensor.eval(session=sess)));
				else:
					sys.stderr.write("		cur batch MSE = %f\n"%(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY})));
			b=0;
			trainingSteps+=1;
			if trainingSteps % args.saveEvery==0:
				if np.isnan(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY})):
					raise Exception("ERROR: reached nan MSE - quitting without saving.");
				save_path = saver.save(sess, args.outFPre+".ckpt")
				saveParams(sess)
				if args.verbose>1:
					sys.stderr.write("	cur session saved in %s\n"%save_path);
		data=line.rstrip().split("\t");
	if args.verbose>1: 
		if args.L1 is not None:
			sys.stderr.write("	cur run MSE = %f;\tloss = %f;\tmodel paramsum = %f;\tparamsnum = %i\n"%(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY}),myLoss.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY}),paramPenaltyL1Tensor.eval(session=sess),paramNumActivityTensor.eval(session=sess)));
		else:
			sys.stderr.write("	cur run MSE = %f\n"%(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY})));
	inFile.close();


#train with remaining data, but remove anything past b
if b > 0:
	batchX = batchX[0:b,:,:,:]
	batchY = batchY[0:b];
	sess.run(train_step, feed_dict={ohcX: batchX, realELY: batchY})
	trainingSteps+=1;
if np.isinf(mseTF.eval(session=sess, feed_dict={ohcX: batchX, realELY: batchY})):
	raise Exception("ERROR: reached nan MSE - quitting without saving.");

save_path = saver.save(sess, args.outFPre+".ckpt")
if args.verbose>1:
	sys.stderr.write("	cur session saved in %s\n"%save_path);

saveParams(sess)

sess.close()
if (args.logFP is not None):
	logFile.close();








