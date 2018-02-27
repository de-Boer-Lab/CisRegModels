#!/usr/bin/env python
import warnings
from CisRegModels import MYUTILS
import sys
import argparse
parser = argparse.ArgumentParser(description='Takes a file containing many promoter sequences and counts them, outputting them in decreasing order of abundance.')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file of promoter sequences', required=True);
parser.add_argument('-o',dest='outFP', metavar='<outFile>',help='Where to output results [default=stdout]', required=False);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);

args = parser.parse_args();


inFile=MYUTILS.smartGZOpen(args.inFP,'r');

promoterCounts = {};

if (args.logFP is not None):
	logFile=MYUTILS.smartGZOpen(args.logFP,'w');
	sys.stderr=logFile;

if (args.outFP is None):
	outFile= sys.stdout;
else:
	if args.verbose>0: warnings.warn("Outputting to file "+args.outFP);
	outFile = MYUTILS.smartGZOpen(args.outFP,'w');

#raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
for line in inFile:
	if line is None or line == "" or line[0]=="#": continue
	line = line.rstrip();
	if line not in promoterCounts:
		promoterCounts[line]=1;
	else:
		promoterCounts[line]+=1;
inFile.close();

for i in reversed(sorted(promoterCounts, key=promoterCounts.__getitem__)):
	outFile.write("%s\t%i\n"%(i, promoterCounts[i]));
outFile.close();
if (args.logFP is not None):
	logFile.close();
