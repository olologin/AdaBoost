library(kernlab)
library(caret)

StochGradientDescent <- function(X, y, W, gradientFunc, costFunc, initParams, learningRate) {
  # Stochastic gdesc, computes gradient of 1 point from dataset and changes
  # weights
  params <- initParams
  pPrev <- initParams - 1
  for (i in 1:1000000){
    indices <- sample(1:nrow(X), 15)
    gradients <- gradientFunc(X[indices,], y[indices], W[indices], params)
    params <- params - learningRate*gradients
    #q <- costFunc(X,y,params)
    #print(q)
    if(abs(sum(pPrev^2) - sum(params^2))<1e-6){
      break
    }
    pPrev <- params
  }
  params
}

logisticRegressionCostFunction <- function(X, y, w, params) {
  mean(w*log2(1+exp(as.vector(X%*%params)*-y)))
}

logisticRegressionGradient <- function(X, y, w, params) {
  colMeans(-(X*y*w)/(1+exp(as.vector(X%*%params)*y)))
}

# Loading and normalization of data
data1 = as.matrix(read.csv("non linear dataset.csv", header = FALSE))

normalizeData <- function(X) {
    for(i in 1:ncol(X)) {
      m <- mean(X[,i])
      X[,i] <- X[,i] - m
      X[,i] <- X[,i] / sd(X[,i])
    }
    X
}

X <- normalizeData(data1[,-3])
l <- nrow(X)

X <- cbind(X, rep(1, l)) # adding artificial feature

y <- data1[,3]

plot(X[,-3], pch=y+20)

learningRate <- l*0.01
T <- 20  # How much elementary algs to learn

# vector with objects weights
W <- rep(1/l, l)
# vector with classifiers weights
A <- rep(0, T)
# matrix where each n-th column contains weight of n-th classifier
B <- matrix(nrow=ncol(X), ncol=T)

for (t in 1:T) {
  initParams <- rep(0, ncol(X))
  params <- StochGradientDescent(X, y, W, logisticRegressionGradient,
                                 logisticRegressionCostFunction,
                                 initParams,
                                 learningRate)
  B[,t] <- params
  abline((-params[3]/params[2]), -params[1]/params[2], col='red')
  h <- X%*%params
  #Eta <- sum(W*exp(-y*h))
  Eta <- sum(W[which((h*y)<0)])
  A[t] <- (1/2)*log(1+((1-Eta)/Eta))
  #Eta <- sum(W[which(h*y<0)])
  #A[t] <- (1/2)*log(((1-Eta)/Eta))
  W <- W*exp(-y*A[t]*h)
  W <- W/sum(W)
  print(W)
}
A <- A/sum(A)
confusionMatrix(c(X%*%B%*%A<0), y<0)