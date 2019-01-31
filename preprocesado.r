library(Hmisc)
library(mice)
require(DMwR) # KNN imp
library(mvoutlier)  
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



findOutliers <- function(col,coef){ 
  cuartil.primero = quantile(col,0.25)
  cuartil.tercero = quantile(col,0.75)
  iqr <- cuartil.tercero - cuartil.primero
  
  extremo.superior.outlier <- cuartil.tercero + coef * iqr
  extremo.inferior.outlier <- cuartil.primero - coef * iqr
  
  return( which((col > extremo.superior.outlier) | (col < extremo.inferior.outlier),arr.ind=TRUE))
}

vector_claves_outliers_IQR_en_alguna_columna <- function(datos, coef=1.5){
  vector.es.outlier <- sapply(datos[1:ncol(datos)], findOutliers,coef)
  vector.es.outlier
}

computeOutliers <- function(data, type='remove', k=2){
  outliers <- unlist(vector_claves_outliers_IQR_en_alguna_columna(data))
  if (type == 'remove'){
    index.to.keep <- setdiff(c(1:nrow(data)),outliers)
    return (data[index.to.keep,])
  }
  else if(type == 'knn'){
    data[outliers,] <- rep(NA,ncol(data))
    return(computeMissingValues(data,type='knn',k=k))
  }
  
  return(data)
}
