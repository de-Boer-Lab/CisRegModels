#!/usr/bin/env python
import warnings
import sys
import argparse
parser = argparse.ArgumentParser(description='Creates a biochemical transcription model using OHC sequence input.')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file of one-hot-code sequences, preceeded by their expression value, tab delim', required=True);
parser.add_argument('-sl',dest='seqLen',	metavar='<seqLen>',help='Input sequence length in bp', required=True);
parser.add_argument('-b',dest='batch',	metavar='<batchSize>',help='Batch size to use to feed into the neural net [default = 1000] ', required=False, default = "1000");
parser.add_argument('-raw',dest='runningAverageWindow',	metavar='<runningAverageWindow>',help='Window to use when reporting the running average MSE per batch [default = 100] ', required=False, default = "100");
parser.add_argument('-se',dest='saveEvery',	metavar='<saveEvery>',help='save neural net every X batches [default = 500] ', required=False, default = "500");
parser.add_argument('-res',dest='restore',	metavar='<restoreFromCkptFile>',help='Restore NN from file ', required=False);
parser.add_argument('-nm',dest='numMotifs',	metavar='<numMotifs>',help='The number of motifs to include in the model [default=400]', required=False, default = "400");
parser.add_argument('-ml',dest='motifLen',	metavar='<motifLen>',help='Motif length [default = 20]', required=False, default = "20");
parser.add_argument('-dm',dest='motifsFP',	metavar='<defaultMotifFile>',help='Pre-load the PWMs with these', required=False);
parser.add_argument('-mll',dest='meanSDFile',	metavar='<meanSDRelationshipFile>',help='If this is specified, instead of the MSE, minimize the negative log liklihood and use this curve to calculate predicted SDs for promoters.', required=False);
parser.add_argument('-ic',dest='initConcs', metavar='<initConcs>',help='Initialize log concentrations to these (usually min Kds for the PWM; same order as others) [default log(concs)=0]', required=False);
parser.add_argument('-ia',dest='initActiv', metavar='<initActiv>',help='Initialize activities to these - can be either a file with values or a float [defaults to gaussian]', required=False);
parser.add_argument('-ip',dest='initPotent', metavar='<initPotent>',help='Initialize potentiations to these - can be either a file with values or a float [defaults to gaussian]', required=False);
parser.add_argument('-eb',dest='useEBound', action='count',help='use expected binding rather than prob binding?', required=False, default=0);
parser.add_argument('-ca',dest='clearAdamVars', action='count',help='clear AdamOptimizer variables?', required=False, default=0);
parser.add_argument('-ae',dest='Aepsilon',help='AdamOptimizer Epsilon value [default = 1E-8]', required=False, default="1E-8");
parser.add_argument('-ab1',dest='Abeta1',help='AdamOptimizer beta1 value [default = 0.9]', required=False, default="0.9");
parser.add_argument('-ab2',dest='Abeta2',help='AdamOptimizer beta2 value [default = 0.999]', required=False, default="0.999");
parser.add_argument('-mo',dest='useMomentumOptimizer', action='count',help='use MomentumOptimizer instead of Adam for SGD optimization?', required=False, default=0);
parser.add_argument('-mon',dest='useNesterov', action='count',help='use Nesterov MomentumOptimizer?', required=False, default=0);
parser.add_argument('-mom',dest='momentum',help='momentum parameter (RMSProp and MomentumOptimizer) [default = 0.0]', required=False, default="0.0");
parser.add_argument('-rms',dest='useRMSProp', action='count',help='use RMSProp instead of Adam for SGD optimization?', required=False, default=0);
parser.add_argument('-rmsd',dest='rmsDecay',help='RMSProp decay parameter [default = 0.9]', required=False, default="0.9");
parser.add_argument('-rmse',dest='rmsEpsilon',help='RMSProp epsilon parameter [default = 1e-10]', required=False, default="1e-10");
parser.add_argument('-bl',dest='bindingLimits', action='count',help='Include a maximum binding parameter?', required=False, default=0);
parser.add_argument('-ibl',dest='initBindLim', metavar='<initBindLim>',help='Initialize binding limits to these [defaults to 1]', required=False);
parser.add_argument('-po',dest='potentiation', action='count',help='add a layer of potentiation, modulating accessibility?', required=False, default=0);
parser.add_argument('-aisa',dest='accIsAct', action='count',help='Include a constant relating openness directly to activity (output as concetration constant)', required=False, default=0);
parser.add_argument('-posa',dest='trainPositionalActivities',action='count',help='Train position specific activity scales (where 1 is no scaling, 0 scales activity to no effect, and -1 flips the sign of the activity? [default=no]. Note that this necessarily makes the model an EBound model after the potentiation step', required=False, default=0);
parser.add_argument('-stra',dest='trainStrandedActivities',action='count',help='Train activities capturing strand-specific differences in activity? [default=no]', required=False, default=0);
parser.add_argument('-ntc',dest='noTrainConcs', action='count',help='make TF concentrations constant (i.e. whatever concentrations are input)?', required=False, default=0);
parser.add_argument('-nta',dest='noTrainActivities', action='count',help='make TF activities constant (i.e. whatever activities are input)?', required=False, default=0);
parser.add_argument('-ntp',dest='noTrainPotentiations', action='count',help='make TF potentiations constant (i.e. whatever potentiations are input)?', required=False, default=0);
parser.add_argument('-ntm',dest='noTrainMotifs', action='count',help='Don\'t change the motif layer of variables?', required=False, default=0);
parser.add_argument('-ntbl',dest='noTrainBL', action='count',help='Don\'t change the binding limits?', required=False, default=0);
parser.add_argument('-r',dest='runs',	metavar='<numRuns>',help='How many times to run through data [default=1]', required=False, default="1");
parser.add_argument('-lr',dest='learningRate',	metavar='<learningRate>',help='the learning rate parameter (bigger = learns faster, but noisier) [default=0.01]', required=False, default="0.01");
parser.add_argument('-lred',dest='learningRateED',	metavar='<learningRateExponentialDecay>',help='Every 175 steps, decay the learning rate by this factor: LR\'=LR*this [good value ==0.99; default= no decay]', required=False, default="1.0");
parser.add_argument('-clr',dest='cycleLearningRate',	metavar='<cycleLearningRate>',help='Parameters for cylcing the learning rate, as follows: LR_low,LR_high,period, where period is half a full cycle', required=False);
parser.add_argument('-l1',dest='L1',	metavar='<l1Penalty>',help='L1-regularization parameter for the activities and potentiations (good values are ~0.00001 [default=no regularization]', required=False);
parser.add_argument('-l2',dest='L2',	metavar='<l2Penalty>',help='L2-regularization parameter for the PWMs (good values are ~0.000001) [default=no regularization]', required=False);
parser.add_argument('-l2Pos',dest='L2Pos',	metavar='<l2PenaltyForPositionalActivity>',help='L2-regularization parameter for differences in adjacent positional activities of TFs (good values are ~0.00001) [default=no regularization]', required=False);
parser.add_argument('-t',dest='threads',	metavar='<threads>',help='Number of threads to make use of [default=1]',default = "1", required=False);
parser.add_argument('-o',dest='outFPre', metavar='<outFilePrefix>',help='Where to output results - prefix [default=stdout]', required=False);
parser.add_argument('-tb',dest='tensorboard', metavar='<tensorboardFile>',help='Output tensorboard summary?', required=False);
parser.add_argument('-tbf',dest='tensorboardFrequency', metavar='<tensorboardFrequency>',help='Output tensorboard every N minibatches [default=log2]', required=False);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);
parser.add_argument('-trace',dest='trace', action='count',help='Run an execution trace and save to outFPre.trace.json?', required=False, default=0);
args = parser.parse_args();

if args.trace >0: 
	import os
	os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/extras/CUPTI/lib64/:"+os.environ['LD_LIBRARY_PATH']
	sys.stderr.write("LD_LIBRARY_PATH = %s\n" % os.environ['LD_LIBRARY_PATH'])

import CisRegModels.MYUTILS
import CisRegModels.PWM;
import CisRegModels;
import tensorflow as tf
import numpy as np;
from datetime import datetime
import os;

args.batch = int(args.batch);
args.seqLen = int(args.seqLen);
if args.tensorboardFrequency is not None:
	args.tensorboardFrequency = int(args.tensorboardFrequency);
args.runs = int(args.runs);
args.numMotifs = int(args.numMotifs);
args.motifLen = int(args.motifLen);
args.saveEvery = int(args.saveEvery);
args.Aepsilon = float(args.Aepsilon);
args.Abeta1 = float(args.Abeta1);
args.Abeta2 = float(args.Abeta2);
args.learningRate = float(args.learningRate);
args.learningRateED = float(args.learningRateED);
args.threads = int(args.threads);
args.runningAverageWindow = int(args.runningAverageWindow);
args.rmsDecay = float(args.rmsDecay);
args.rmsEpsilon = float(args.rmsEpsilon);

myModel = CisRegModels.SETUPOHCENHANCOSOMEMODEL.CRM(args);
myModel.makeModel();

