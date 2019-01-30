library(Hmisc)
library(mice)
require(DMwR) # KNN imp
set.seed(1234)


# MISSING DATA
fillNAMean <- function(col){
  col[is.na(col)] <- mean(col, na.rm = TRUE)
  return(col)
}

computeMissingValues <- function(data, type='remove',k=2) {
  if(anyNA(data)){
    if (type == 'remove') data <- data[complete.cases(data),]
    else if (type == 'mean'){
      data[,1:dim(data)[2]] <- sapply(data[,1:dim(data)[2]], fillNAMean)
    }
    else if(type == 'knn'){
      data <- knnImputation(data,k=k)
    }
  }
  return(data)
}

data <- read.csv("./train.csv", header=TRUE, na.strings="?")

# CENTER AND SCALE DATA
require(caret)
n <- length(data)
valoresPreprocesados <- caret::preProcess(data[1:n-1],method=c("center" ,"scale") )
data.scaled <- predict(valoresPreprocesados,data[1:n-1])
data.scaled <- cbind(data.scaled,C=data$C)

train.data <- computeMissingValues(data.scaled,type='knn')