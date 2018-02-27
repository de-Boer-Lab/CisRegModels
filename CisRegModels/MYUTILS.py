import gzip

def smartGZOpen(filename,mode):
	if len(filename)>3 and filename[-3:].lower()=='.gz':
		return gzip.open(filename,'%st'%(mode));
	else:
		return open(filename,mode);

def smartGZForeach(filename):
	inFile = smartGZOpen(filename,"r")
	for line in inFile:
		yield line

