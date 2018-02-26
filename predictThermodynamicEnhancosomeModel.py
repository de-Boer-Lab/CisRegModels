#!/usr/bin/env python
import warnings
import sys
import argparse
parser = argparse.ArgumentParser(description='Predicts biochemical transcription model on new data using OHC sequence input.')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file of one-hot-code sequences, preceeded by their expression value, tab delim', required=True);
parser.add_argument('-sl',dest='seqLen',	metavar='<seqLen>',help='Input sequence length in bp', required=True);
parser.add_argument('-b',dest='batch',	metavar='<batchSize>',help='Batch size to use to feed into the neural net [default = 1000] ', required=False, default = "1000");
parser.add_argument('-M',dest='loadModel',	metavar='<restoreSavedModel>',help='Restore graph and parameters from file', required=False);
parser.add_argument('-VARIABLE',dest='VARIABLE',action='count', help='Makes the name of the concentration tensor "Variable" - for backward compatibility', required=False);
parser.add_argument('-res',dest='restore',	metavar='<restoreFromCkptFile>',help='Restore parameters from file ', required=False);
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


import SETUPOHCENHANCOSOMEMODEL;
myCRM = SETUPOHCENHANCOSOMEMODEL.CRM(args);
myCRM.testModel()


