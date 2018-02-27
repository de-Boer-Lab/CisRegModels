#!/usr/bin/env python
import warnings
import CisRegModels.MYUTILS
import CisRegModels.MYMATH
import numpy as np
import subprocess
import sys
import argparse
parser = argparse.ArgumentParser(description='Provided a file of sequences and priorities (sorted), uses Bowtie to merge similar sequences.')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file of sequences and priorities, tab delimited 2 columns', required=True);
parser.add_argument('-t',dest='tempFilePre',	metavar='<tempFilePre>',help='Path prefix where to store temp files for Bowtie DB (etc).', required=True);
parser.add_argument('-th',dest='threads',	metavar='<numThreads>',help='Number of Bowtie threads to use [default = 1]', default = "1", required=False);
parser.add_argument('-o',dest='outFPre', metavar='<outFilePrefix>',help='Where to output results, prefix', required=True);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);
parser.add_argument('-sd',dest='skipDB', action='count',help='Skip the database creation step (e.g. was already done)?', required=False, default=0);
parser.add_argument('-bbp',dest='bowtieBuildParams',help='additional parameters (quoted) for bowtie build.', required=False, default="");
parser.add_argument('-sa',dest='skipAlignment', action='count',help='Skip the alignment step (e.g. was already done)? - also skip DB creation', required=False, default=0);

args = parser.parse_args();

verbose = args.verbose;

if (args.logFP is not None):
	logFile=MYUTILS.smartGZOpen(args.logFP,'w');
	sys.stderr=logFile;

#test if bowtie exists
p=subprocess.call(["which","bowtie2"], stdin=subprocess.PIPE, stderr=subprocess.PIPE);
if p!=0:
	raise Exception("could not find bowtie2.  Did you use Bowtie2 ?");

outFileMap = MYUTILS.smartGZOpen(args.outFPre + "_map.txt.gz",'w');
outFileCounts = MYUTILS.smartGZOpen(args.outFPre + "_counts.txt.gz",'w');

if verbose>0:
	sys.stderr.write("Inputting data...");
inFile=MYUTILS.smartGZOpen(args.inFP,'r');
(promoterSeqs, colNames, promoterCounts) = MYMATH.inputMatrix(inFile, inType = np.int, colnames=False, rownames=True)
inFile.close();
if verbose>0:
	sys.stderr.write(" done!\n");

if args.bowtieBuildParams!="":
	sys.stderr.write("Using additional bowtie2-build parameters '%s'..."%args.bowtieBuildParams);

def makeBowtieDB():
	global args;
	global promoterSeqs;
	fastaOut = MYUTILS.smartGZOpen(args.tempFilePre + ".seqs.fasta",'w');
	for i in range(0,len(promoterSeqs)):
		fastaOut.write(">%i\n%s\n"%(i,promoterSeqs[i]));
	fastaOut.close();
	#subprocess.check_call(["bowtie2-build","%s.seqs.fasta"%args.tempFilePre,"%s.bowtie2"%args.tempFilePre], stdout=subprocess.PIPE, stdin=subprocess.PIPE);
	p = subprocess.Popen(["bowtie2-build"] + args.bowtieBuildParams.split() + ["%s.seqs.fasta"%args.tempFilePre,"%s.bowtie2"%args.tempFilePre], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE);
	if args.verbose>0:
		sys.stderr.write("Running bowtie2-build...");
	(curStdout, stderrData) = p.communicate();
	if args.verbose>0:
		sys.stderr.write(curStdout);
		sys.stderr.write(stderrData);
		sys.stderr.write("done!\n");



if args.skipDB==0 and args.skipAlignment==0:
	if verbose>0:
		sys.stderr.write("Making Bowtie DB...");
	makeBowtieDB();
	if verbose>0:
		sys.stderr.write("done!\n");

if args.skipAlignment==0:
	if verbose>0:
		sys.stderr.write("Running Auto-alignment...");
	bowtieCommand = ["bowtie2","-N","1","-L","18","-a","-p",args.threads,"-f", "--no-sq", "--no-head", "--un","%s.unaligned"%args.tempFilePre, "-x", "%s.bowtie2"%args.tempFilePre, "-U", "%s.seqs.fasta"%args.tempFilePre, "-S", "%s.hits.sam"%args.tempFilePre ];
	if verbose>0:
		sys.stderr.write(" ".join(bowtieCommand));
	p = subprocess.Popen(bowtieCommand, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE);
	(curStdout, stderrData) = p.communicate();
	if args.verbose>0:
		sys.stderr.write(curStdout);
		sys.stderr.write(stderrData);
		sys.stderr.write("done!\n");

if verbose>0:
	sys.stderr.write("Reading Auto-alignment...");
bowtieOut = open("%s.hits.sam"%args.tempFilePre, "r")
observed2source = list(range(0,len(promoterSeqs))); # initially set the srouces to identity.
lastRead=0;
for line in bowtieOut:
	#qname, flag, rname, pos, mapq, cigar, rnext, pnext, tlen, seq, qual
	data = line.rstrip().split("\t");
	qname = int(data[0]);
	if data[2]=="*":
		sys.stderr.write("\nRead %i did not map to anything: %s"%(qname,promoterSeqs[qname]))
	else:
		rname = int(data[2]);
		if observed2source[qname] > observed2source[rname]: #if rname is a higher priority than whatever qname is set to
			 observed2source[qname] = observed2source[rname]; #set the source to whatever the source for rname is
		if  observed2source[rname] > observed2source[qname]: # opposite
			 observed2source[rname] = observed2source[qname]; #set the source to whatever the source for qname is

if verbose>0:
	sys.stderr.write("done!\nCombining counts...");
for i in range(0, len(promoterSeqs)):
	if observed2source[i]!=i:
		promoterCounts[observed2source[i]] = promoterCounts[observed2source[i]] + promoterCounts[i];
		promoterCounts[i]=0;
if verbose>0:
	sys.stderr.write("done!\nOutputting results...");
nMerged =0;
for i in [x for (c,x) in sorted(zip(promoterCounts, list(range(0, len(promoterSeqs)))), reverse=True)]: #output in decreasing order of the new sequence counts
	if observed2source[i] == i: # was not merged with something else
		outFileCounts.write("%s\t%i\n"%(promoterSeqs[i], promoterCounts[i]));
	else:
		nMerged+=1;
		outFileMap.write("%s\t%s\n"%(promoterSeqs[i], promoterSeqs[observed2source[i]])); #from -> to

outFileMap.close();
outFileCounts.close();
if verbose>0:
	sys.stderr.write("all done!\n");
sys.stderr.write("Merged %i sequences total\n"%nMerged);

if (args.logFP is not None):
	logFile.close();
