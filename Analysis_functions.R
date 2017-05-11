
treeHyperGeo = function(clustering, annotation, minN=5){
  #clustering=TFPositionalClustering ; annotation=testAnnots
  annotData = data.frame(id = clustering$label, i=1:length(clustering$label), annotation = annotation, stringsAsFactors=F)
  treeStats = data.frame(i=1:nrow(clustering$merge), n=0, a=0)
  for (i in 1:nrow(clustering$merge)){
    if (clustering$merge[i,1]<0){
      treeStats$n[i] = treeStats$n[i]+1;
      treeStats$a[i] = treeStats$a[i]+annotData$a[-clustering$merge[i,1]]
    }else{
      treeStats$n[i] = treeStats$n[i]+treeStats$n[clustering$merge[i,1]];
      treeStats$a[i] = treeStats$a[i]+treeStats$a[clustering$merge[i,1]]
    }
    if (clustering$merge[i,2]<0){
      treeStats$n[i] = treeStats$n[i]+1;
      treeStats$a[i] = treeStats$a[i]+annotData$a[-clustering$merge[i,2]]
    }else{
      treeStats$n[i] = treeStats$n[i]+treeStats$n[clustering$merge[i,2]];
      treeStats$a[i] = treeStats$a[i]+treeStats$a[clustering$merge[i,2]]
    }
  }
  n=treeStats$a[nrow(treeStats)]; #white balls
  m=treeStats$n[nrow(treeStats)] - treeStats$a[nrow(treeStats)]; # black balls
  treeStats$logP = NA
  for (i in 1:nrow(treeStats)){
    if (treeStats$n[i] >= minN && (length(clustering$labels) - treeStats$n[i]) >= minN){ #enough points in branch and rest
      treeStats$logP[i] = phyper(treeStats$n[i]-treeStats$a[i], m, n, treeStats$n[i], log.p = T)
    }
  }
  return(treeStats)
}
multiTreeHyperGeo = function(clustering, annotations, minN=5){
  allTreeStats=data.frame()
  annotations = annotations[as.character(clustering$labels),]
  for(i in 1:ncol(annotations)){
    curHG = treeHyperGeo(clustering,annotations[,i],minN=minN)
    curHG$annotation=names(annotations)[i];
    allTreeStats = rbind(allTreeStats, curHG)
  }
  allTreeStats$P = exp(allTreeStats$logP)
  allTreeStats$FDR = p.adjust(allTreeStats$P, n = sum(!is.na(allTreeStats$logP)), method="BH")
  return(allTreeStats);
}



revcompPFM = function(x){
  newPFM = x;
  newPFM$A=x$T
  newPFM$T=x$A
  newPFM$G=x$C
  newPFM$C=x$G
  newPFM = newPFM[order(newPFM$pos,decreasing=T),];
  newPFM$pos = 1:nrow(newPFM);
  return(newPFM);
}

padPFM = function(x, nBefore=0, nAfter = 0){
  blankRow = x[1,]
  blankRow$A[1]=0.25;blankRow$C[1]=0.25;blankRow$G[1]=0.25;blankRow$T[1]=0.25;
  newPFM = data.frame();
  if (nBefore>0){
    for (i in 1:nBefore){
      newPFM = rbind(newPFM, blankRow);
    }
  }
  newPFM = rbind(newPFM, x);
  if (nAfter>0){
    for (i in 1:nAfter){
      newPFM = rbind(newPFM, blankRow);
    }
  }
  newPFM$pos = 1:nrow(newPFM);
  return(newPFM);
}

mergePFMs = function(pfmA, pfmB, offset, useIC=F){
  if (offset > 0){
    pfmA = padPFM(pfmA, nBefore=offset)
  }else if (offset < 0){
    pfmB = padPFM(pfmB, nBefore=-offset)
  }
  if (nrow(pfmA) > nrow(pfmB)){
    pfmB = padPFM(pfmB, nAfter=nrow(pfmA) - nrow(pfmB))
  }else if (nrow(pfmA) < nrow(pfmB)){
    pfmA = padPFM(pfmA, nAfter=nrow(pfmB) - nrow(pfmA))
  }
  if(useIC){
      for (i in 1:nrow(pfmA)){
      pfmA$A[i] = (pfmA$A[i]*pfmA$IC[i] + pfmB$A[i]*pfmB$IC[i])/ (pfmA$IC[i] + pfmB$IC[i]);
      pfmA$C[i] = (pfmA$C[i]*pfmA$IC[i] + pfmB$C[i]*pfmB$IC[i])/ (pfmA$IC[i] + pfmB$IC[i]);
      pfmA$G[i] = (pfmA$G[i]*pfmA$IC[i] + pfmB$G[i]*pfmB$IC[i])/ (pfmA$IC[i] + pfmB$IC[i]);
      pfmA$T[i] = (pfmA$T[i]*pfmA$IC[i] + pfmB$T[i]*pfmB$IC[i])/ (pfmA$IC[i] + pfmB$IC[i]);
    }
  }else{
    for (i in 1:nrow(pfmA)){
      pfmA$A[i] = (pfmA$A[i] + pfmB$A[i])/2;
      pfmA$C[i] = (pfmA$C[i] + pfmB$C[i])/2;
      pfmA$G[i] = (pfmA$G[i] + pfmB$G[i])/2;
      pfmA$T[i] = (pfmA$T[i] + pfmB$T[i])/2;
    }
  }
  pfmA$pos = 1:nrow(pfmA);
  return(pfmA);
}

trimPFM = function(pfmA, minIC=0.1){
  trimmedFront = T;
  while (trimmedFront){
    if (nrow(pfmA)>0){
      curIC = log2(pfmA$A[1]/0.25)*pfmA$A[1] + log2(pfmA$C[1]/0.25)*pfmA$C[1] + log2(pfmA$G[1]/0.25)*pfmA$G[1] +log2(pfmA$T[1]/0.25)*pfmA$T[1]
      if (curIC< minIC){
        pfmA = pfmA[2:nrow(pfmA),];
      }else{
        trimmedFront=F
      }
    }else{trimmedFront=F}
  }
  trimmedBack = T;
  while (trimmedBack){
    if (nrow(pfmA)>0){
      curIC = log2(pfmA$A[nrow(pfmA)]/0.25)*pfmA$A[nrow(pfmA)] + log2(pfmA$C[nrow(pfmA)]/0.25)*pfmA$C[nrow(pfmA)] + log2(pfmA$G[nrow(pfmA)]/0.25)*pfmA$G[nrow(pfmA)] +log2(pfmA$T[nrow(pfmA)]/0.25)*pfmA$T[nrow(pfmA)]
      if (curIC< minIC){
        pfmA = pfmA[1:(nrow(pfmA)-1),];
      }else{
        trimmedBack=F
      }
    }else{trimmedBack=F}
  }
  pfmA$pos = 1:nrow(pfmA);
  return(pfmA);
}
