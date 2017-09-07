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
parser.add_argument('-bl',dest='bindingLimits', action='count',help='Include a maximum binding parameter?', required=False, default=0);
parser.add_argument('-ibl',dest='initBindLim', metavar='<initBindLim>',help='Initialize binding limits to these [defaults to 1]', required=False);
parser.add_argument('-po',dest='potentiation', action='count',help='add a layer of potentiation, modulating accessibility?', required=False, default=0);
parser.add_argument('-aisa',dest='accIsAct', action='count',help='Include a constant relating openness directly to activity (output as concetration constant)', required=False, default=0);
parser.add_argument('-tftf',dest='interactions', metavar='<windowSize>',help='Add a layer of interaction filters to scale Kds based on nearby Kds within this window [default = do not use interaction terms]', required=False, default=0);
parser.add_argument('-posa',dest='trainPositionalActivities',action='count',help='Train slopes capturing a linear positional dependence? [default=no]. Note that this necessarily makes the model an EBound model after the potentiation step', required=False, default=0);
parser.add_argument('-stra',dest='trainStrandedActivities',action='count',help='Train activities capturing strand-specific differences in activity? [default=no]', required=False, default=0);
parser.add_argument('-ntc',dest='noTrainConcs', action='count',help='make TF concentrations constant (i.e. whatever concentrations are input)?', required=False, default=0);
parser.add_argument('-nta',dest='noTrainActivities', action='count',help='make TF activities constant (i.e. whatever activities are input)?', required=False, default=0);
parser.add_argument('-ntp',dest='noTrainPotentiations', action='count',help='make TF potentiations constant (i.e. whatever potentiations are input)?', required=False, default=0);
parser.add_argument('-ntm',dest='noTrainMotifs', action='count',help='Don\'t change the motif layer of variables?', required=False, default=0);
parser.add_argument('-ntbl',dest='noTrainBL', action='count',help='Don\'t change the binding limits?', required=False, default=0);
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


if tf.__version__=='1.2.1':
	with open('/home/unix/cgdeboer/lib/python/SETUPOHCENHANCOSOMEMODEL.tf.1.2.1.py') as fd:
		exec(fd.read())
else:
	with open('/home/unix/cgdeboer/lib/python/SETUPOHCENHANCOSOMEMODEL.py') as fd:
		exec(fd.read())


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

