import math
import numpy
from CisRegModels import MYUTILS

def inputMatrix(inFile, inType = numpy.float, colnames=True, rownames=True):
	colNames = []
	nRows = 0;
	nCols = 0;
	maxString=1;
	first=colnames;
	for line in inFile:
		if line is None or line=="": continue;
		data = line.rstrip().split("\t");
		if first:
			first=False;
			colNames = data;
		else:
			nCols = len(data);
			if inType==numpy.str:
				for i in range(1,len(data)):
					maxString = max(maxString,len(data[i]))
			nRows+=1;
	if rownames:
		nCols = nCols-1;
		rowLabs = [""] * nRows
	else:
		rowLabs = [];
	if inType==numpy.str:
		dataMatrix = numpy.empty((nRows,nCols), dtype = numpy.dtype('|S%i'%(maxString)));
	else:
		dataMatrix = numpy.empty((nRows,nCols), dtype = inType) 
	nRows = 0;
	first=colnames;
	inFile.seek(0,0);
	for line in inFile:
		if line is None or line=="": continue;
		data = line.rstrip().split("\t");
		if first:
			first=False;
		else:
			if rownames:
				rowLabs[nRows] = data[0];
				dataMatrix[nRows,:] = numpy.array([data[1:len(data)]]).astype(inType)
			else:
				dataMatrix[nRows,:] = numpy.array([data[0:len(data)]]).astype(inType)
			nRows+=1;
	return (rowLabs, colNames, dataMatrix);

def saveMatrix(outFileName, rowLabs, colLabs, dataMatrix):
	outFile = MYUTILS.smartGZOpen(outFileName, "w");
	outFile.write("\t".join(colLabs)+"\n");
	for i in range(0,len(rowLabs)):
		outFile.write(rowLabs[i]);
		for j in range(0,dataMatrix.shape[1]):
			outFile.write("\t%g"%dataMatrix[i,j]);
		outFile.write("\n");
	outFile.close();

def scaleLength(data,desiredLength):
	scaleMiddle(data,0,length(data),desiredLength);




def scaleMiddle(data,fStart,fEnd,desiredLength,func="MEAN"): #from start to end-1
	#now scale middle bin accordingly to desiredLength.
	thisMid = reshapeData(data, fStart, fEnd, desiredLength, func);
	#print(data[:fStart]);
	#print(thisMid)
	scaledData = list(data[:fStart]) + thisMid + list(data[fEnd+1:]);
	return scaledData;

def isfloat(x):
	return issubclass(type(x), numpy.float) or isinstance(x,float)

def reshapeData(curProfile,fStart,fEnd,desiredLength,func):
	#combines data by average
	#print(fStart)
	#print(fEnd)
	#print(desiredLength)
	curLen = fEnd-fStart+1;
	#print(curLen)
	thisMid = [0.0]*desiredLength;
	theCounts = [0]*desiredLength;
	mapRatio=(0.0+desiredLength)/curLen;
	#print(mapRatio)
	if curLen>=desiredLength:#compress
		for j in range(0,curLen):
			if ((fStart+j)>=len(curProfile) or math.isnan(curProfile[fStart+j]) or not numpy.isreal(curProfile[fStart+j])):
				continue  #skip if data has a NaN
			dest=(mapRatio*j);
			ceilPct =0;
			floor=int(math.floor(dest));
			ceil=int(math.floor(dest+mapRatio));
			if ceil>floor:
				ceilPct=(dest+mapRatio- math.floor(dest+mapRatio))/mapRatio;
			floorPct=1-ceilPct;
			#print([curLen, desiredLength, dest, mapRatio, floorPct, ceilPct, floor, ceil].join("\t")+"\n");
			#p [fStart, j, floorPct, floor];
			#p curProfile;

			thisMid[floor]+=floorPct*curProfile[fStart+j];
			theCounts[floor]+=floorPct;
			if ceil<desiredLength:
				thisMid[ceil]+=ceilPct*curProfile[fStart+j];
				theCounts[ceil]+=ceilPct;
	else: #expand
		for j in range(0,curLen):
			if ((fStart+j)>=len(curProfile) or  math.isnan(curProfile[fStart+j]) or not numpy.isreal(curProfile[fStart+j])):
				continue;
			dest = (mapRatio*(j));
			nextDest = dest+mapRatio;
			first=int(math.floor(dest))
			last=int(math.floor(nextDest))
			totalPct=0.0;
			for l in range(first,last+1):
				if l==desiredLength:
					continue;
				elif l==first:
					curPct = (1-(dest-math.floor(dest)))/mapRatio;
				elif l==last:
					curPct = (nextDest-math.floor(nextDest))/mapRatio;
				else:
					curPct=1.0/mapRatio;
				#print([curLen, desiredLength, dest, nextDest, mapRatio, first, last, l, curPct].join("\t")+"\n");
				theCounts[l]+=curPct;
				thisMid[l]+=curPct*curProfile[fStart+j];
				totalPct+=curPct;
			if abs(totalPct-1.0)>=0.001:
				print("ERROR: Total Percent is not 1: "+totalPct.to_s()+"\n");
	for i in range(0, len(thisMid)): # Replace those without data withno data with NaN and average the rest
		if theCounts[i]==0:
			thisMid[i]=float('NaN');
		elif func=="MEAN":
			thisMid[i] = thisMid[i]/theCounts[i];
	return thisMid;

def Angle2D(x1, y1, x2, y2):
   theta1 = np.arctan2(y1,x1);
   theta2 = np.arctan2(y2,x2);
   dtheta = theta2 - theta1;
   while (dtheta > np.pi):
      dtheta -= 2*np.pi;
   while (dtheta < -np.pi):
      dtheta += 2*np.pi;
   return(dtheta);

def isInPolygon(polygonX, polygonY, pointsX, pointsY):
    allInside = [];
    for i in range(0,len(pointsX)):
        angle=0;
        for j in range(0,len(polygonX)):
            angle += Angle2D(polygonX[j]-pointsX[i], polygonY[j]-pointsY[i],polygonX[(j+1)%len(polygonX)]-pointsX[i], polygonY[(j+1)%len(polygonX)]-pointsY[i]);
        if (np.abs(angle) >= np.pi):
            allInside.append(i)
    return allInside;
#
#def reshapeDataAvg(curProfile,fStart,fEnd,desiredLength):
#	#combines data by average
#	curLen = fEnd-fStart
#	thisMid = [0.0]*desiredLength;
#	theCounts = [0.0]*desiredLength;
#	mapRatio=(0.0+desiredLength)/curLen;
#	if curLen>=desiredLength:#compress
#		for j in range(0,curLen):
#			dest=(mapRatio*(j));
#			ceilPct =0;
#			floor=dest.floor();
#			ceil=(dest+mapRatio).floor();
#			if ceil>floor:
#				ceilPct=(dest+mapRatio- (dest+mapRatio).floor())/mapRatio;
#			floorPct=1-ceilPct;
#			#print([curLen, desiredLength, dest, mapRatio, floorPct, ceilPct, floor, ceil].join("\t")+"\n");
#			#p [cPIndex, j, floorPct, floor];
#			#p curProfile;
#			if curProfile[cPIndex+j]==NaN  or  curProfile[cPIndex+j] is None:
#				continue  #skip if data has a NaN
#
#			thisMid[floor]+=floorPct*curProfile[cPIndex+j];
#			theCounts[floor]+=floorPct;
#			if ceil<desiredLength:
#				thisMid[ceil]+=ceilPct*curProfile[cPIndex+j];
#				theCounts[ceil]+=ceilPct;
#	else: #expand
#		for j in range(0,curLen):
#			dest = (mapRatio*(j));
#			nextDest = dest+mapRatio;
#			first=dest.floor();
#			last=(nextDest).floor()
#			totalPct=0.0;
#			first.upto(last){|l|
#				if l==desiredLength:
#					continue;
#				elif l==first:
#					curPct = (1-(dest-dest.floor()))/mapRatio;
#				elif l==last:
#					curPct = (nextDest-nextDest.floor())/mapRatio;
#				else:
#					curPct=1.0/mapRatio;
#				#print([curLen, desiredLength, dest, nextDest, mapRatio, first, last, l, curPct].join("\t")+"\n");
#				if (curProfile[cPIndex+j] is None  or  curProfile[cPIndex+j]==NaN):
#					continue;
#				theCounts[l]+=curPct;
#				thisMid[l]+=curPct*curProfile[cPIndex+j];
#				totalPct+=curPct;
#			if totalPct-1.0>=0.001:
#				print("ERROR: Total Percent is not 1: "+totalPct.to_s()+"\n");
#	for i in range(0, thisMid.length()): # Replace those without data withno data with NaN and average the rest
#		if theCounts[i]==0:
#			thisMid[i]=NaN;
#		else:
#			thisMid[i] = thisMid[i]/theCounts[i];
#	return thisMid;
#
#def reshapeDataSum(curProfile,curLen, cPIndex):
#	#combines data by sum -unsure if this is actually working correctly because it's a weird method - I think it mght also average
#	curLen = fEnd-fStart
#	thisMid = [0.0]*desiredLength;
#	mapRatio=(0.0+desiredLength)/curLen;
#	if curLen>=desiredLength: #compress
#		for j in range(0,curLen):
#			dest=(mapRatio*(j));
#			ceilPct =0;
#			floor=dest.floor();
#			ceil=(dest+mapRatio).floor();
#			if ceil>floor:
#				ceilPct=(dest+mapRatio- (dest+mapRatio).floor())/mapRatio;
#			floorPct=1-ceilPct;
#			#print([curLen, desiredLength, dest, mapRatio, floorPct, ceilPct, floor, ceil].join("\t")+"\n");
#			#p [cPIndex, j, floorPct, floor];
#			#p curProfile;
#			if curProfile[cPIndex+j]==NaN: #skip if data has a NaN
#				continue
#			#p([curProfile.length(),j,cPIndex, curLen]);
#			thisMid[floor]+=floorPct*curProfile[cPIndex+j];
#			if ceil<desiredLength:
#				thisMid[ceil]+=ceilPct*curProfile[cPIndex+j];
#	else: #expand
#		for j in range(0,curLen):
#			dest = (mapRatio*(j));
#			nextDest = dest+mapRatio;
#			first=dest.floor();
#			last=(nextDest).floor()
#			totalPct=0.0;
#			first.upto(last){|l|
#				if l==desiredLength:
#					continue;
#				elif l==first:
#					curPct = (1-(dest-dest.floor()))/mapRatio;
#				elif l==last:
#					curPct = (nextDest-nextDest.floor())/mapRatio;
#				else:
#					curPct=1.0/mapRatio;
#				#print([curLen, desiredLength, dest, nextDest, mapRatio, first, last, l, curPct].join("\t")+"\n");
#				if curProfile[cPIndex+j]!=NaN:
#					thisMid[l]+=curPct*curProfile[cPIndex+j];
#				totalPct+=curPct;
#			if totalPct-1.0>=0.001:
#				print("ERROR: Total Percent is not 1: "+totalPct.to_s()+"\n");
#	return thisMid;
