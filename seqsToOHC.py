#!/usr/bin/env python
import warnings
import MYUTILS
import sys
import argparse
parser = argparse.ArgumentParser(description='Converts a set of sequences into a one-hot-code (binary) representation - excludes non [ATGC] chars.  Output in ACGT order, one line per sequence, base then position.')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file of sequences with a value in the second column that will preceed the OHC output on each line, separated by a tab', required=True);
parser.add_argument('-m',dest='maxLen',	metavar='<maxSeqLen>',help='The maximum sequence length to consider (truncated after this point)', required=True);
parser.add_argument('-b',dest='orientBack', action='count',help='Align sequences of different sizes to back [default=front]?', required=False, default=0);
parser.add_argument('-o',dest='outFP', metavar='<outFile>',help='Where to output results [default=stdout]', required=False);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);

args = parser.parse_args();


inFile=MYUTILS.smartGZOpen(args.inFP,'r');
maxSeqLen = int(args.maxLen);

if (args.logFP is not None):
	logFile=MYUTILS.smartGZOpen(args.logFP,'w');
	sys.stderr=logFile;

if (args.outFP is None):
	outFile= sys.stdout;
else:
	if args.verbose>0: sys.stderr.write("Outputting to file "+args.outFP+"\n");
	outFile = MYUTILS.smartGZOpen(args.outFP,'w');
BASES = ['A','C','G','T'];
#raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
for line in inFile:
	if line is None or line == "" or line[0]=="#": continue
	data=line.rstrip().split("\t");
	curSeq = data[0].upper();
	curLabel = data[1];
	curSeqLen = len(curSeq);
	outFile.write(curLabel+"\t");
	for b in BASES:
		if args.orientBack>0: #output the front until we run out of chars, then print 0s, or truncate if it's longer than maxSeqLen
			if curSeqLen > maxSeqLen:
				curSeq = curSeq[:maxSeqLen];
			outFile.write("\t".join([ "1" if curSeq[i]==b else "0" for i in range(0,len(curSeq))]));
			if maxSeqLen > curSeqLen: # print 0s
				outFile.write("\t" + "\t".join(["0"]*(maxSeqLen - curSeqLen)));
		else: #print any 0s at the front and also truncate from the front if too long.
			if maxSeqLen > curSeqLen: # print 0s
				outFile.write("\t".join(["0"]*(maxSeqLen - curSeqLen)) + "\t");
			elif curSeqLen > maxSeqLen:
				curSeq = curSeq[(len(curSeq) - maxSeqLen):];
			outFile.write("\t".join([ "1" if curSeq[i]==b else "0" for i in range(0,len(curSeq))]));
		if b!="T":
			outFile.write("\t");
	outFile.write("\n");
inFile.close();
outFile.close();
if (args.logFP is not None):
	logFile.close();
