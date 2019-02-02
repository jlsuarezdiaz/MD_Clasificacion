library(Hmisc)
library(mice)
require(DMwR) # KNN imp
library(mvoutlier)  
library(randomForest)
library(FSelector)
library(Boruta)
library(NoiseFiltersR)

set.seed(1234)

# MISSING DATA
fillNAMean <- function(col){
  col[is.na(col)] <- mean(col, na.rm = TRUE)
  return(col)
}

fillNAMedian <- function(col){
  col[is.na(col)] <- median(col, na.rm = TRUE)
  return(col)
}

changeOutliersValue <- function(outliers,data,type = 'median'){
  i = 1
  n = ncol(data)
  # no funciona bien, siguen saliendo outliers.
  while(i <= n){
    outliers_columna = outliers[[i]]
    data[outliers_columna,i] = mean(data[,i], na.rm = TRUE)
    i = i +1
  }
  
  return(data)
}

computeMissingValues <- function(data, type='remove',k=2) {
  if(anyNA(data)){
    if (type == 'remove') data <- data[complete.cases(data),]
    else if (type == 'mean'){
      data[,1:dim(data)[2]] <- sapply(data[,1:dim(data)[2]], fillNAMean)
    }
    else if (type == 'median'){
      data[,1:dim(data)[2]] <- sapply(data[,1:dim(data)[2]], fillNAMedian)
    }
    else if(type == 'knn'){
      data <- knnImputation(data,k=k)
    }
    else if(type == 'rf'){
      data <- rfImpute(data[1:length(data)-1], data[,length(data)], iter = 5, tree = 100)
      class = data[,1]
      data = data[,-1]
      data = cbind(data, C = class)
    }
    else if(type == 'mice'){
      tempData <- mice(data, m = 5, meth="pmm", maxit = 50, seed = 500)
      data = complete(tempData,1)
    }
  }
  return(data)
}

# CENTER AND SCALE DATA
require(caret)

preProcessData <- function(data,test){
  n <- length(data)
  valoresPreprocesados <- caret::preProcess(data[1:n-1],method=c("center" ,"scale") )
  data.scaled <- predict(valoresPreprocesados,data[1:n-1])
  data.scaled <- cbind(data.scaled,C=data$C)
  
  # realizamos el mismo preprocesado para el test según la media y varianza de cada columna del train
  means = apply(data[1:n-1],2,mean)
  sds = apply(data[1:n-1],2,sd)
  test.scaled = as.data.frame(scale(test, center = means, scale = sds))
  
  list(data.scaled,test.scaled)
}


findOutliers <- function(col,coef = 1.5){ 
  cuartil.primero = quantile(col,0.25)
  cuartil.tercero = quantile(col,0.75)
  iqr <- cuartil.tercero - cuartil.primero
  
  extremo.superior.outlier <- cuartil.tercero + coef * iqr
  extremo.inferior.outlier <- cuartil.primero - coef * iqr
  
  return( which((col > extremo.superior.outlier) | (col < extremo.inferior.outlier),arr.ind=TRUE))
}

vector_claves_outliers_IQR_en_alguna_columna <- function(datos, coef=1.5){
  vector.es.outlier.normal <- sapply(datos[1:ncol(datos)], findOutliers,coef)
  vector.es.outlier.normal
}


computeOutliers <- function(data, type='remove', k=2){
  outliers <- vector_claves_outliers_IQR_en_alguna_columna(data)

  if (type == 'remove'){
    index.to.keep <- setdiff(c(1:nrow(data)),unlist(outliers))
    return (data[index.to.keep,])
  }
  else if(type == 'knn'){
    data[unlist(outliers),] <- rep(NA,ncol(data))
    return(computeMissingValues(data,type='knn',k=k))
  }
  else if(type == 'median'){
    return(changeOutliersValue(outliers,data))
  }
  else if(type == 'mean'){
    data[outliers,] <- rep(NA,ncol(data))
    return(computeMissingValues(data,type='mean',k=k))
  }
  else if(type == 'rf'){
    data[outliers,] <- rep(NA,ncol(data))
    return(computeMissingValues(data,type='rf',k=k))
  }
  else if(type == 'mice'){
    data[outliers,] <- rep(NA,ncol(data))
    return(computeMissingValues(data,type='mice',k=k))
  }
  
  return(data) # es necesario?
}

featureSelection <- function(method,number, data, Class){
  if (method == 'chi'){
    weights <- FSelector::chi.squared(Class~., data)
    subset <- FSelector::cutoff.k(weights,number)
  }
  else if(method == 'lc'){
    weights <- FSelector::linear.correlation(Class~., data)
    subset <- FSelector::cutoff.k(weights,number)
  }
  else if(method == 'rc'){
    weights <- FSelector::rank.correlation(Class~., data)
    subset <- FSelector::cutoff.k(weights,number)
  }
  else if(method == 'ig'){
    weigths <- FSelector::information.gain(Class~., data)
    subset <- FSelector::cutoff.k(weights,number)
  }
  else if(method == 'gr'){
    weigths <- FSelector::gain.ratio(Class~., data)
    subset <- FSelector::cutoff.k(weights,number)    
  }
  else if(method == 'su'){
    weigths <- FSelector::symmetrical.uncertainty(Class~., data)
    subset <- FSelector::cutoff.k(weights,number)     
  }
  else if(method == 'oneR'){
    weights <- FSelector::oneR(Class~.,data)
    subset <- FSelector::cutoff.k(weights,number)
  }
  else if(method == 'relief'){
    weights <- FSelector::relief(Class~., data, neighbours.count = 5, sample.size = 20)
    subset <- FSelector::cutoff.k(weights,number)
  }
  else if(method == 'cfs'){
    subset <- FSelector::cfs(Class~.,data)
  }
  else if(method == 'cons'){
    subset <- FSelector::consistency(Class~.,data)
  }
  else if(method == 'rfi'){
    weights <- FSelector::random.forest.importance(Class~.,data, importance.type = 1)
    subset <- FSelector::cutoff.k(weights,number)
  }
  
  return(subset)
}

removeHighCorrelationAttributes <- function(data,umbral){
  tmp <- cor(data)
  tmp[!lower.tri(tmp)] <-0
  data.new <- data[,!apply(tmp,2,function(x) any(x > umbral))]
  return(data.new)
}

computeImportanceAttributes <- function(datos,Class){
  set.seed(7)
  control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 5)
  modelo <- caret::train(Class~.,data = datos, methdod = "lvq", trControl = control)
  importance <- caret::varImp(modelo, scale = FALSE)
  
  return(importance)
}

rankingLearningRandomForest <- function(data,Class){
  set.seed(7)
  control <- caret::rfeControl(functions = rfFuncs, method = "cv", number = 10)
  results <- caret::rfe(data,Class, sizes=c(0:1), rfeControl = control)
  print(results)
  return(predictors(results))
}

applyBoruta <- function(datos,Class){
  Boruta.data <- Boruta(Class~.,data = datos, doTrace = 2)
  print(Boruta.data)
  print(Boruta.data$finalDecision)
  return(Boruta.data)
}

RandomForestAndBoruta <- function(datos,Class){
  Boruta.data = applyBoruta(datos,Class)
  model1 <- randomForest(Class~., data = datos)
  model2 <- randomForest(datos[, getSelectedAttributes(Boruta.data)],Class)
  print(model2)
  plot(Boruta.dara)
}

filterNoiseData <- function(data){
  out.data <- NoiseFiltersR::IPF(data, nfolds = 5, consensus = FALSE, p = 0.01, s = 3, y = 0.5)
  data.clean = out.data$cleanData
  return (data.clean)
}