#!/usr/bin/env python
import warnings
import MYUTILS
import sys
import argparse
parser = argparse.ArgumentParser(description='Translates the contents of a file from one thing to another using a dictionary.')
parser.add_argument('-is',dest='inFPSeqs',	metavar='<inFileSeqs>',help='Input file of sequences to translate.', required=True);
parser.add_argument('-id',dest='inFPDict',	metavar='<inFileDictionary>',help='Input file of translations to use in the form from\tto', required=True);
parser.add_argument('-o',dest='outFP', metavar='<outFile>',help='Where to output results [default=stdout]', required=False);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);

args = parser.parse_args();


inFileDict=MYUTILS.smartGZOpen(args.inFPDict,'r');
inFileSeqs=MYUTILS.smartGZOpen(args.inFPSeqs,'r');


if (args.logFP is not None):
	logFile=MYUTILS.smartGZOpen(args.logFP,'w');
	sys.stderr=logFile;

if (args.outFP is None):
	outFile= sys.stdout;
else:
	if args.verbose>0: sys.stderr.write("Outputting to file "+args.outFP+"\n");
	outFile = MYUTILS.smartGZOpen(args.outFP,'w');

translationDict = {};
#raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
for line in inFileDict:
	if line is None or line == "" or line[0]=="#": continue
	data=line.rstrip().split("\t");
	translationDict[data[0]] = data[1];
inFileDict.close();

for line in inFileSeqs:
	if line is None or line == "" or line[0]=="#": continue
	seq=line.rstrip();
	if seq in translationDict:
		outFile.write(translationDict[seq]+"\n");
	else:
		outFile.write(seq + "\n");
inFileSeqs.close();

outFile.close();

if (args.logFP is not None):
	logFile.close();
