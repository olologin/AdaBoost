library(kernlab)
library(caret)

weightedMode <- function(y, W) {
  posIdx <- which(y==unique(y)[1])
  negIdx <- which(y!=unique(y)[1])
  if(sum(W[posIdx]) > sum(W[negIdx])) {
    return (unique(y)[1])
  } else {
    return (unique(y)[2])
  }
}

entropy <- function(W, y) {
  posIdx <- which(y==unique(y)[1])
  negIdx <- which(y!=unique(y)[1])
  posProb <- sum(W[posIdx])/sum(W)
  negProb <- sum(W[negIdx])/sum(W)
  result <- 0
  if(posProb > 0) {
    result = result + posProb*log2(posProb)
  }
  if(negProb > 0) {
    result = result + negProb*log2(negProb)
  }
  -result
}

gain <- function(X, W, y) {
  bestColNumber <- 1
  bestSep <- 1
  bestGain <- -100
  H <- entropy(W, y)

  for (colNumber in 1:ncol(X)) {
    column <- X[,colNumber]
    dataSize <- length(column)
    sorted <- sort(column, index.return=TRUE)
    for (sepN in 1:(dataSize-1)) {
      #weightedSum <- (sepN/dataSize) * entropy(W[sorted$ix[1:sepN]], y[sorted$ix[1:sepN]]) +
      #((dataSize-sepN)/dataSize)*entropy(W[sorted$ix[(sepN+1):dataSize]], y[sorted$ix[(sepN+1):dataSize]])
      
      # случай когда признаки равны, мы не можем разделить их, пропускаем
      if(sorted$x[sepN]==sorted$x[sepN+1])
        next
      
      weightedSum <- (sum(W[sorted$ix[1:sepN]])/sum(W)) * entropy(W[sorted$ix[1:sepN]], y[sorted$ix[1:sepN]]) +
        (sum(W[sorted$ix[(sepN+1):dataSize]])/sum(W))*entropy(W[sorted$ix[(sepN+1):dataSize]], y[sorted$ix[(sepN+1):dataSize]])
      gain <- H-weightedSum
      if (gain > bestGain) {
        bestGain <- gain
        bestColNumber <- colNumber
        bestSep <- sorted$x[sepN] + ((sorted$x[sepN+1] - sorted$x[sepN])/2)
      }
    }
  }
  list(colNumber=bestColNumber, sep=bestSep, gain=bestGain)
}

create_dtree <- function(X, W, y, depth) {
  X <- as.matrix(X)
  mValue <- weightedMode(y, W)
  if(length(unique(y))==1 || depth==0) {
    return (list(isLeaf=TRUE, class=mValue))
  } else {
    partition <- gain(X, W, y)
    
    leftIdx <- which(X[,partition$colNumber] <= partition$sep)
    rightIdx <- which(X[,partition$colNumber] > partition$sep)
    
    return (list(isLeaf=FALSE,
                 l=create_dtree(X[leftIdx, ], W[leftIdx], y[leftIdx], depth-1),
                 r=create_dtree(X[rightIdx, ], W[rightIdx], y[rightIdx], depth-1),
                 colNumber=partition$colNumber,
                 sep=partition$sep))
    
  }
}

classify_dtree_singleX <- function(dtree, singleX) {
  if(dtree$isLeaf) {
    return (dtree$class)
  }
  if(singleX[dtree$colNumber] > dtree$sep) {
    return (classify_dtree_singleX(dtree$r, singleX))
  } else {
    return (classify_dtree_singleX(dtree$l, singleX))
  }
}
classify_dtree <- function(dtree, X) {
  y <- c()
  for (i in 1:nrow(X)) {
    y <- c(y, classify_dtree_singleX(dtree, X[i,]))
  }
  y
}


# Loading and normalization of data
data1 = as.matrix(read.csv("non linear dataset.csv", header = FALSE))
#data1 = as.matrix(read.csv("ex2data1.txt", header = FALSE))

normalizeData <- function(X) {
  for(i in 1:ncol(X)) {
    m <- mean(X[,i])
    X[,i] <- X[,i] - m
    X[,i] <- X[,i] / sd(X[,i])
  }
  X
}

#X <- normalizeData(data1[,-3])
X <- data1[,-3]
l <- nrow(X)

y <- data1[,3]

y[which(y==0)]=-1

plot(X, pch=y+20)

T <- 20  # How much elementary algs to learn

# vector with objects weights
W <- rep(1/l, l)
# vector with classifiers weights
A <- rep(0, T)
# matrix where each n-th column contains weight of n-th classifier
B <- list()

for (t in 1:T) {
  dtree <- create_dtree(X[,], W, y, 4)
  B[[t]] <- dtree
  h <- classify_dtree(dtree, X)
  idx <- which(h*y<0)
  #Eta <- sum(W*exp(-y*h))
  Eta <- sum(W[idx])
  if(Eta > 0.5){
    T <- t-1
    break
  }
  #A[t] <- (1/2)*log(1+((1-Eta)/Eta))
  A[t] <- (1/2)*log(((1-Eta)/Eta))
  #A[t] <- log(((1-Eta)/Eta))+log(1)
  #W <- W*(exp(A[t]*-y*h))#/(sqrt((1-Eta)/Eta))
  W[idx] <- W[idx]*exp(-y[idx]*A[t]*h[idx])
  W <- W/sum(W)
  #print(W)
}

s = rep(0, length(y))
for(t in 1:T) {
  s = s + (A[t]*classify_dtree(B[[t]], X))
}
points(X[which((s<0)!=(y<0)),], pch=35, col="red")

#A <- A/sum(A)
confusionMatrix(s<0, y<0)