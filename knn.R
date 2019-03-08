train <- read.csv("train.csv", na.strings = c("?", "NA", "NR", "na", "NaN", "nan"))
test <- read.csv("test.csv", na.strings = c("?", "NA", "NR", "na", "NaN", "nan"))

## BIBLIOTECAS

library(ggplot2)
library(caret)
library(RKEEL)
library(kknn)
library(GGally)
library(Hmisc)
library(dplyr)
library(corrplot)
library(tidyr)
library(VIM)
library(mice)
library(bmrm)
library(DMwR)
library(NoiseFiltersR)
library(beeswarm)
library(moments)
library(MASS)
library(FSelector)

# Función que crea los archivos de sumisión.
createSubmission <- function(pred, filename){
  sub <- cbind(Id = 1:length(pred), Prediction = as.numeric(as.character(pred)))
  write.csv(sub, paste0("subs-knn/",filename), row.names = F)
  sub
}

# ----------------------------------------------------------------------------------------- #
# SUBIDA 1
funcion.train.predict <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  indices.nas.train <- which(has.na(train))
  knn.model <- train(C ~ ., train[-c(outliers.train.por.la.cara, indices.nas.train),], method="knn", preProcess = c("center", "scale"))
  
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.prueba <- funcion.train.predict(train, test)

sub <- createSubmission(sub.prueba, "prueba") 


# ---------------------------------------------------------------------------------------- #
# SUBIDA 2
train.predict.nas.por.media <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  # Imputación de NAs por defecto Mice
  imputados <- mice::mice(train, m=1, method="mean")
  train.completed <- mice::complete(imputados)
  knn.model <- train(C ~ ., train.completed, method="knn", preProcess = c("center", "scale"))
  
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.prueba2 <- train.predict.nas.por.media(train, test)

createSubmission(sub.prueba2, "prueba2")


# ------------------------------------------------------------------------------------- #
# SUBIDA 3
# Imputación de NAs con knn
train.predict.knn.imputation <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  knn.model <- train(C ~ ., train.completed, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.3 <- train.predict.knn.imputation(train, test)
createSubmission(sub.3, "3")


# ----------------------------------------------------------------------------------- #
# SUBIDA 4
# Imputación de NAs con knn por clases
train.predict.knn.imputation.by.class <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  knn.model <- train(C ~ ., train.completed, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.4 <- train.predict.knn.imputation.by.class(train, test)
createSubmission(sub.4, "4")

# -------------------------------------------------------------------------------- #
# SUBIDA 5
# Imputación knn + PCA al 95 %
train.predict.pca <- function(train, test, ncomps){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train)
  pca <- prcomp(train.completed[,-ncol(train.completed)], center=T, scale=T)
  train.pca <- as.data.frame(predict(pca, train.completed)[,1:ncomps])
  knn.model <- train(C ~ ., cbind(train.pca,C = train.completed$C), method="knn", tuneGrid = expand.grid(k=c(15)))
  print(knn.model)
  # Predict
  test.pca <- as.data.frame(predict(pca, test)[, 1:ncomps])
  preds <- predict(knn.model, test.pca)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.5 <- train.predict.pca(train, test, 30)
createSubmission(sub.5, "5")


# --------------------------------------------------------------------------------- #
# SUBIDA 7
# Imputación knn encadenada
train.predict.knn.imputation.mas.precisa <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- data.frame(train)
  nas.ordenados <- order(apply(train, 2, function(x) sum(is.na(x))), decreasing = T)[-51]
  
  # Imputación knn encadenada
  for(i in 1:50){
    indices.nas.train <- which(has.na(train.completed))
    print(length(indices.nas.train))
    na.labels.train <- train.completed[-indices.nas.train,nas.ordenados[i]]
    na.data.train <- train.completed[-indices.nas.train, -c(nas.ordenados[i], 50)]
    na.data.test <- train.completed[which(is.na(train[,nas.ordenados[i]])), -c(nas.ordenados[i], 50)]
    knn.na.model <- train(na.data.train, na.labels.train, 
                          method="knn", preProcess=c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
    na.labels.test <- predict(knn.na.model, na.data.test)
    train.completed[which(is.na(train[,nas.ordenados[i]])), nas.ordenados[i]] <- na.labels.test
  }
  knn.model <- train(C ~ ., train.completed, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.7 <- train.predict.knn.imputation.mas.precisa(train, test)
createSubmission(sub.7, "7")


# ------------------------------------------------------------------------------ #
# SUBIDA 8
# Imputación knn + EF por consenso
train.predict.knn.imputation.ef <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- EF(train.completed)$cleanData
  knn.model <- train(C ~ ., train.cleaned, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.8 <- train.predict.knn.imputation.ef(train, test)
createSubmission(sub.8, "8")



# ------------------------------------------------------------------------------- #
# SUBIDA 9
# Imputación knn + EF por mayoría
train.predict.knn.imputation.ef.majority <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- EF(train.completed, consensus = F)$cleanData
  knn.model <- train(C ~ ., train.cleaned, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}


set.seed(28)
sub.9 <- train.predict.knn.imputation.ef.majority(train, test)
createSubmission(sub.9, "9")


# ------------------------------------------------------------------------------------ #
# SUBIDA 10
# Imputación knn + CVCF por consenso
train.predict.knn.imputation.cvcf <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = T)$cleanData
  knn.model <- train(C ~ ., train.cleaned, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}


set.seed(28)
sub.10 <- train.predict.knn.imputation.cvcf(train, test)
createSubmission(sub.10, "10")


# ------------------------------------------------------------------------------- #
# SUBIDA 11
# Imputación knn + CVCF por mayoría (este filtro es el que parece funcionar mejor)
train.predict.knn.imputation.cvcf.majority <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  knn.model <- train(C ~ ., train.cleaned, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.11 <- train.predict.knn.imputation.cvcf.majority(train, test)
createSubmission(sub.11, "11")


# ---------------------------------------------------------------------------- #
# SUBIDA 12
# Imputación knn + IPF por consenso
train.predict.knn.imputation.ipf <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- IPF(train.completed, consensus = T)$cleanData
  knn.model <- train(C ~ ., train.cleaned, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.12 <- train.predict.knn.imputation.ipf(train, test)
createSubmission(sub.12, "12")


# ------------------------------------------------------------------------ #
# SUBIDA 13
# Imputación knn + IPF por mayoría
train.predict.knn.imputation.ipf.majority <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- IPF(train.completed, consensus = F)$cleanData
  knn.model <- train(C ~ ., train.cleaned, method="knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.13 <- train.predict.knn.imputation.ipf.majority(train, test)
createSubmission(sub.13, "13")


# ---------------------------------------------------------------------- #
# SUBIDA 14
# 11 + simetrías
train.predict.14 <- function(train, test){
  # Train
  # Fuera outliers por la cara
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  # Imputación knn
  train.completed <- knnImputation(train) 
  # Filtro de ruido
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  # Train
  knn.model <- train(C ~ ., train.cleaned, method="knn", preProcess = c("YeoJohnson", "center", "scale"), 
                     tuneGrid = expand.grid(k=c(1,3,5,7,9,11,13,15)))
  print(knn.model)
  # Predict
  preds <- predict(knn.model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.14 <- train.predict.14(train, test)
createSubmission(sub.14, "14")


# ------------------------------------------------------------------------ #
# SUBIDA 15
# 11 + tuneado de ks y kernels (k = 17, kernel=epanechnikov)
train.predict.15 <- function(train, test, k, kernel){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  scaler <- preProcess(train.cleaned) # Centrado y escalado
  train.scaled <- predict(scaler, train.cleaned)
  knn.model <- train.kknn(C ~ ., train.scaled, ks = k, kernel = kernel, scale=F)
  # Predict
  test.scaled <- predict(scaler, test)
  preds <- predict(knn.model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.15 <- train.predict.15(train, test, 17, "epanechnikov")
createSubmission(sub.15, "15") # 0.91168


# -------------------------------------------------------------------- #
# SUBIDA 16
# 11 + tuneado de ks y kernels (k = 17, kernel=triangular)
set.seed(28)
sub.16 <- train.predict.15(train, test, 17, "triangular")
createSubmission(sub.16, "16") # 0.91220


# -------------------------------------------------------------------- #
# SUBIDA 17
# Lo mismo pero con voto mayoritario entre los mejores kernels
set.seed(28)
sub.17.rect <- train.predict.15(train, test, 17, "rectangular")
sub.17.tri  <- train.predict.15(train, test, 17, "triangular")
sub.17.epa  <- train.predict.15(train, test, 17, "epanechnikov")
sub.17.inv  <- train.predict.15(train, test, 17, "inv")
sub.17.gaus <- train.predict.15(train, test, 17, "gaussian")
sub.17.all <- data.frame(sub.17.rect, sub.17.tri, sub.17.epa, sub.17.inv, sub.17.gaus)
sub.17 <- apply(sub.17.all, 1, function(x) ifelse(sum(x==1) > 2,1,0))
createSubmission(sub.17, "17")


# ------------------------------------------------------------------- #
# SUBIDA 18
# Relief + kernel gaussiano
train.predict.18 <- function(train, test, k, kernel){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  scaler <- preProcess(train.cleaned) # Centrado y escalado
  train.scaled <- predict(scaler, train.cleaned)
  # Pesos relief
  relief.weights <- relief(C ~ ., train.scaled, neighbours.count = 17)$attr_importance
  barplot(relief.weights)
  relief.weights[relief.weights < 0] <- 0
  relief.weights <- sqrt(relief.weights) # Para que funcionen como pesos con la distancia euclidea.
  train.weighted <- data.frame(t(relief.weights * t(train.scaled[,-ncol(train.scaled)])), C = train.scaled$C)
  train.weighted <- train.weighted[, colSums(train.weighted != 0) > 0]
  
  # Train
  knn.model <- train.kknn(C ~ ., train.weighted, ks = k, kernel = kernel, scale=F)
  # Predict
  # Centrado y escalado
  test.scaled <- predict(scaler, test)
  # Pesos relief
  test.weighted <- data.frame(t(relief.weights * t(test.scaled)))
  test.weighted <- test.weighted[, colSums(test.weighted != 0) > 0]
  preds <- predict(knn.model, test.weighted)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.18 <- train.predict.18(train, test, 17, "gaussian")
createSubmission(sub.18, "18")


# ------------------------------------------------------------------------------- #
# SUBIDA 19
# Relief + kernel coseno
set.seed(28)
sub.19 <- train.predict.18(train, test, 17, "cos")
createSubmission(sub.19, "19")


# ------------------------------------------------------------------------------ #
# SUBIDA 20
# Relief + kernel triangular
set.seed(28)
sub.20 <- train.predict.18(train, test, 17, "triangular")
createSubmission(sub.20, "20")

# --------------------------------------------------------------------- #
# SUBIDA 22
# 16 + PCA 99 %
train.predict.22 <- function(train, test, k, kernel, thresh){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  scaler <- preProcess(train.cleaned, method=c("center", "scale", "pca"), thresh=thresh) # escalado + PCA
  train.scaled <- predict(scaler, train.cleaned)
  knn.model <- train.kknn(C ~ ., train.scaled, ks = k, kernel = kernel, scale=F)
  # Predict
  test.scaled <- predict(scaler, test)
  preds <- predict(knn.model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}


set.seed(28)
sub.22 <- train.predict.22(train, test, 17, "gaussian", 0.99)
createSubmission(sub.22, "22")

# ------------------------------------------------------------------------------ #
# SUBIDA 23

train.predict.21 <- function(train, test, k, kernel, ndims){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  scaler <- preProcess(train.cleaned, method=c("center", "scale", "pca"), pcaComp=ndims) # escalado + PCA
  train.scaled <- predict(scaler, train.cleaned)
  knn.model <- train.kknn(C ~ ., train.scaled, ks = k, kernel = kernel, scale=F)
  # Predict
  test.scaled <- predict(scaler, test)
  preds <- predict(knn.model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

# 16 + PCA tuneado (20 dimensiones, gaussian)
set.seed(28)
sub.23 <- train.predict.23(train, test, 17, "gaussian", 20)
createSubmission(sub.23, "23")


# ------------------------------------------------------------------- #
# 16 + outliers (k = 23, kernel=cos)
train.predict.24 <- function(train, test, k, kernel){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  indices.outliers <- which(
    train$X1 > 1000    |
      train$X7 > 300     |
      train$X15 > 600    |
      train$X16 > 400000 |
      train$X17 > 250    |
      train$X20 > 300    |
      train$X21 < -1300  |
      train$X24 > 1700   |
      train$X26 < -1500  |
      train$X29 > 39     |
      train$X33 > 480    |
      train$X39 > 400    |
      train$X43 > 2000   |
      train$X45 > 25
  )
  if(length(indices.outliers) > 0) train <- train[-indices.outliers,]
  print(paste0("Eliminados ",length(indices.outliers), " outliers."))
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  scaler <- preProcess(train.cleaned) # Centrado y escalado
  train.scaled <- predict(scaler, train.cleaned)
  knn.model <- train.kknn(C ~ ., train.scaled, ks = k, kernel = kernel, scale=F)
  # Predict
  test.scaled <- predict(scaler, test)
  preds <- predict(knn.model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}


set.seed(28)
sub.24 <- train.predict.24(train, test, 23, "cos")
createSubmission(sub.24, "24")


# ------------------------------------------------------------------------ #
# SUBIDA 25
# 16 + outliers (k = 23, kernel=gaussian)
set.seed(28)
sub.25 <- train.predict.24(train, test, 23, "gaussian")
createSubmission(sub.25, "25")


# ------------------------------------------------------------------------ #
# SUBIDA 26
# 16 + menos outliers (k=23, kernel=triangular)
train.predict.26 <- function(train, test, k, kernel){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  indices.outliers <- which(
    train.cleaned$X1 > 1000   |
      train.cleaned$X3 > 300000 |
      train.cleaned$X7 > 300    |
      train.cleaned$X38 > 400   |
      train.cleaned$X43 > 2000  
  )
  if(length(indices.outliers) > 0) train.cleaned <- train.cleaned[-indices.outliers,]
  print(paste0("Eliminados ",length(indices.outliers), " outliers."))
  scaler <- preProcess(train.cleaned) # Centrado y escalado
  train.scaled <- predict(scaler, train.cleaned)
  knn.model <- train.kknn(C ~ ., train.scaled, ks = k, kernel = kernel, scale=F)
  # Predict
  test.scaled <- predict(scaler, test)
  preds <- predict(knn.model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.26 <- train.predict.26(train, test, 23, "triangular")
createSubmission(sub.26, "26")

# ------------------------------------------------------------------------------------- #
# SUBIDA 27
# 16 + transformaciones (k=23, kernel=gaussian)
wrong.remove <- function(data){
  transf <- data.frame(data)
  wrongs <- which(
    transf$X7 <= 0
  )
  if(length(wrongs) > 0){
    transf <- transf[-wrongs,]
  }
  transf
}

attr.transform <- function(data){
  trans <- wrong.remove(data)
  trans$X2 <- log(trans$X2)
  trans$X3 <- log(trans$X3)
  trans$X4 <- log(trans$X4)
  trans$X6[trans$X6 < 0] <- 0
  trans$X7 <- log(trans$X7)
  trans$X8 <- trans$X8^2
  trans$X9 <- log(trans$X9)
  trans$X13 <- trans$X13^2
  trans$X15 <- log(trans$X15)
  trans$X16 <- log(trans$X16)
  trans$X17 <- log(trans$X17)
  trans$X18 <- log(trans$X18)
  # trans$X21 <- cbrt(trans$X21)
  train$X23[train$X23 < 0] <- 0
  trans$X23 <- log(trans$X23 + 100)
  trans$X24 <- log(trans$X24)
  trans$X25 <- trans$X25^2
  trans$X26 <- cbrt(trans$X26)
  trans$X27 <- cbrt(trans$X27)
  trans$X28 <- log(trans$X28)
  trans$X29 <- log(trans$X29)
  trans$X31 <- log(trans$X31)
  trans$X33 <- log(trans$X33)
  trans$X34 <- trans$X34^2
  trans$X35[trans$X35 < 0] <- 0
  trans$X35 <- log(trans$X35 + 0.1)
  trans$X39 <- log(trans$X39)
  trans$X40 <- cbrt(trans$X40)
  trans$X43 <- cbrt(trans$X43)
  trans$X44 <- log(trans$X44)
  trans$X45[trans$X45 < 0] <- 0 
  trans$X45 <- sqrt(trans$X45)
  trans$X47 <- cbrt(trans$X47)
  trans$X48 <- log(trans$X48 + 25)
  trans$X49 <- cbrt(trans$X49)
  trans
}

attr.transform.add <- function(data){
  trans <- wrong.remove(data)
  
  trans$tX2 <- log(trans$X2)
  trans$tX3 <- log(trans$X3)
  trans$tX4 <- log(trans$X4)
  trans$X6[trans$X6 < 0] <- 0
  trans$tX7 <- log(trans$X7)
  trans$tX8 <- trans$X8^2
  trans$tX9 <- log(trans$X9)
  trans$tX13 <- trans$X13^2
  trans$tX15 <- log(trans$X15)
  trans$tX16 <- log(trans$X16)
  trans$tX17 <- log(trans$X17)
  trans$tX18 <- log(trans$X18)
  trans$X21 <- cbrt(trans$X21)
  train$X23[train$X23 < 0] <- 0
  trans$tX23 <- log(trans$X23 + 100)
  trans$tX24 <- log(trans$X24)
  trans$tX25 <- trans$X25^2
  trans$tX26 <- cbrt(trans$X26)
  trans$tX27 <- cbrt(trans$X27)
  trans$tX28 <- log(trans$X28)
  trans$tX29 <- log(trans$X29)
  trans$tX31 <- log(trans$X31)
  trans$tX33 <- log(trans$X33)
  # trans$tX34 <- trans$X34^2
  trans$X35[trans$X35 < 0] <- 0
  trans$tX35 <- log(trans$X35 + 0.1)
  # trans$tX39 <- log(trans$X39)
  trans$tX40 <- cbrt(trans$X40)
  # trans$tX43 <- cbrt(trans$X43)
  trans$tX44 <- log(trans$X44)
  trans$X45[trans$X45 < 0] <- 0 
  trans$tX45 <- sqrt(trans$X45)
  trans$tX47 <- cbrt(trans$X47)
  trans$tX48 <- log(trans$X48 + 25)
  trans$tX49 <- cbrt(trans$X49)
  
  ##
  trans <- trans[,-c(11,34,39,41,43)]
  # trans <- trans[,-c(34)]
  # trans <- trans[,-c(11,34,39,41,43)]
  # trans <- trans[, -c(11,34,39,41,43,12,13,19,50)]
  ##
  
  trans
}

train.predict.27 <- function(train, test, k, kernel){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  train.transformed <- attr.transform(train.cleaned)
  scaler <- preProcess(train.transformed) # Centrado y escalado
  train.scaled <- predict(scaler, train.transformed)
  knn.model <- train.kknn(C ~ ., train.scaled, ks = k, kernel = kernel, scale=F)
  # Predict
  test[outliers.test.por.la.cara,] <- 1
  test.transformed <- attr.transform(test)
  test.scaled <- predict(scaler, test.transformed)
  preds <- predict(knn.model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.27 <- train.predict.27(train, test, 23, "gaussian")
createSubmission(sub.27, "27")

# --------------------------------------------------------------- #
# SUBIDA 28
# 16 + transformaciones (k=17, kernel=triangular)
set.seed(28)
sub.28 <- train.predict.27(train, test, 17, "triangular")
createSubmission(sub.28, "28")

# --------------------------------------------------------------- #
# SUBIDA 29
# 16 +  transformaciones añadidas
train.predict.29 <- function(train, test, k, kernel){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  train.transformed <- attr.transform.add(train.cleaned) 
  scaler <- preProcess(train.transformed) # Centrado y escalado
  train.scaled <- predict(scaler, train.transformed)
  knn.model <- train.kknn(C ~ ., train.scaled, ks = k, kernel = kernel, scale=F)
  # Predict
  test[outliers.test.por.la.cara,] <- 1
  test.transformed <- attr.transform.add(test)
  test.scaled <- predict(scaler, test.transformed)
  preds <- predict(knn.model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.29 <- train.predict.29(train, test, 25, "optimal")
createSubmission(sub.29, "29")


# ------------------------------------------------- #
# SUBIDA 30
# 16 +  transformaciones añadidas
set.seed(28)
sub.30 <- train.predict.29(train, test, 17, "triangular")
createSubmission(sub.30, "30")

# ------------------------------------------------- #
# SUBIDA 31
# Seleccionando transformaciones (todas menos X11, X34, X39, X41, X43) k=17, ker=triangular
set.seed(28)
sub.31 <- train.predict.29(train, test, 17, "triangular")
createSubmission(sub.31, "31") # 0.91424

# ------------------------------------------------- #
# SUBIDA 32
# Seleccionando transformaciones (todas menos X11, X34, X39, X41, X43) k=25 ker=triangular
set.seed(28)
sub.32 <- train.predict.29(train, test, 25, "triangular")
createSubmission(sub.32, "32")

# ------------------------------------------------- #
# SUBIDA 33
# Seleccionando transformaciones (todas menos X11, X34, X39, X41, X43) k=19 ker=epanechnikov
set.seed(28)
sub.33 <- train.predict.29(train, test, 19, "epanechnikov")
createSubmission(sub.33, "33") # 0.91475

# ------------------------------------------------- #
# SUBIDA 34
# Distancias con DMLMJ
library(rDML)
train.predict.dmlmj <- function(train, test, k, kernel){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  train.transformed <- attr.transform.add(train.cleaned) %>% dplyr::select(-C,C)
  scaler <- preProcess(train.transformed) # Centrado y escalado
  train.scaled <- predict(scaler, train.transformed)
  dmlmj <- dml$DMLMJ(n_neighbors = 5)
  dmlmj$fit(train.scaled[,-ncol(train.scaled)], train.scaled[,ncol(train.scaled)])
  train.dmlmj <- data.frame(dmlmj$transform(train.scaled[,-ncol(train.scaled)]), C=train.scaled$C)
  knn.model <- train.kknn(C ~ ., train.dmlmj, ks = k, kernel = kernel, scale=F)
  # Predict
  test[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
  test.transformed <- attr.transform.add(test)
  test.scaled <- predict(scaler, test.transformed)
  test.dmlmj <- data.frame(dmlmj$transform(test.scaled))
  preds <- predict(knn.model, test.dmlmj)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
sub.34 <- train.predict.dmlmj(train, test, 19, "epanechnikov")
createSubmission(sub.34, "34")

# ------------------------------------------------- #
# SUBIDA 35
# Seleccionando transformaciones (X34) k=19 ker=epanechnikov (hay que modificar la función attr.transform.add descomentando la transformación indicada, igual en las futuras subidas)
set.seed(28)
sub.35 <- train.predict.29(train, test, 19, "epanechnikov")
createSubmission(sub.35, "35")


# -------------------------------------------------- #
# SUBIDA 36
# Seleccionando transformaciones (las de 33 sin cambiar variables a 0) k=19 ker=epanechnikov
set.seed(28)
sub.36 <- train.predict.29(train, test, 19, "epanechnikov")
createSubmission(sub.36, "36")


# -------------------------------------------------- #
## SUBIDA 38
# Seleccionando transformaciones (quitando todas con chi2 < 0.2) k=19 ker=epanechnikov
set.seed(28)
sub.38 <- train.predict.29(train, test, 19, "epanechnikov")
createSubmission(sub.38, "38") 


# -------------------------------------------------- #
# SUBIDA 39
# Cambio de orden en preprocesamiento: CVCF -> transformaciones -> imputación knn
k <- 19
kernel <- "epanechnikov"
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
knn.model <- train.kknn(C ~ ., train.scaled.2, ks = k, kernel = kernel, scale=F)

test[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
test.transformed <- attr.transform.add(test)
test.scaled <- predict(scaler, test.transformed)
preds <- predict(knn.model, test.scaled)
preds[outliers.test.por.la.cara] <- 0
sub.39 <- preds
createSubmission(sub.39, "39")


# ----------------------------------------------------- #
# SUBIDA 41
# 39, con k=17 y kernel triangular
k <- 17
kernel <- "triangular"
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
knn.model <- train.kknn(C ~ ., train.scaled.2, ks = k, kernel = kernel, scale=F)

test[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
test.transformed <- attr.transform.add(test)
test.scaled <- predict(scaler, test.transformed)
preds <- predict(knn.model, test.scaled)
preds[outliers.test.por.la.cara] <- 0
sub.41 <- preds
createSubmission(sub.41, "41")


# ---------------------------------------------------------- #
# SUBIDA 42
# 39 + k=23, kernel=gaussian.
k <- 23
kernel <- "gaussian"
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
knn.model <- train.kknn(C ~ ., train.scaled.2, ks = k, kernel = kernel, scale=F)

test[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
test.transformed <- attr.transform.add(test)
test.scaled <- predict(scaler, test.transformed)
preds <- predict(knn.model, test.scaled)
preds[outliers.test.por.la.cara] <- 0
sub.42 <- preds
createSubmission(sub.42, "42")


# ----------------------------------------------------------- #
## SUBIDA 43
# 39 + k=17, kernel=epanechnikov
k <- 17
kernel <- "epanechnikov"
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
knn.model <- train.kknn(C ~ ., train.scaled.2, ks = k, kernel = kernel, scale=F)

testt <- data.frame(test)
testt[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
test.transformed <- attr.transform.add(testt)
test.scaled <- predict(scaler, test.transformed)
preds <- predict(knn.model, test.scaled)
preds[outliers.test.por.la.cara] <- 0
sub.43 <- preds
createSubmission(sub.43, "43") # 0.91832


# ----------------------------------------------------------- #
# SUBIDA 44
# Lo anterior + aprendiendo distancias con NCA
set.seed(28)
k <- 17
kernel <- "epanechnikov"
library(bmrm)
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
X <- train.scaled.2[,-ncol(train.scaled.2)]
y <- train.scaled.2[, ncol(train.scaled.2)]
partitions <- balanced.cv.fold(train.scaled.2$C, num.cv = 9) # Particionamos los datos en nueve muestras igualmente distribuidas (si no, el algoritmo es demasiado lento).
train.predict.nca <- function(i, train){
  partition.indexes <- which(partitions==i)
  Xtra <- train.scaled.2[partition.indexes, -ncol(train.scaled.2)]
  ytra <- train.scaled.2[partition.indexes, ncol(train.scaled.2)]
  
  # Aprendizaje de distancias
  nca = dml$NCA()
  nca$fit(Xtra, ytra)
  X.transformed <- nca$transform(X)
  train.dml <- data.frame(X.transformed, C = y)
  names(train.dml) <- names(train.transformed.2)
  # K-NN
  knn.model.2 <- train.kknn(C ~ ., train.dml, ks = k, kernel = kernel, scale=F)
  # Predicción
  testt <- data.frame(test)
  testt[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
  test.transformed <- attr.transform.add(testt) 
  test.scaled <- predict(scaler, test.transformed)
  test.dml <- nca$transform(test.scaled)
  test.dml <- as.data.frame(test.dml)
  
  names(test.dml) <- names(test.transformed)
  preds <- predict(knn.model.2, test.dml)
  preds[outliers.test.por.la.cara] <- 0
  preds
}

sub.50.all <- sapply(1:9, train.predict.nca, train = train.scaled.2)
sub.50 <- apply(sub.50.all, 1, function(x) ifelse(sum(x==1) > 4,1,0))
createSubmission(sub.50, "50")


# -------------------------------------------------------------- #
# SUBIDA 46
# 39 + k=15, kernel=epanechnikov
k <- 15
kernel <- "epanechnikov"
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
knn.model <- train.kknn(C ~ ., train.scaled.2, ks = k, kernel = kernel, scale=F)

testt <- data.frame(test)
testt[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
test.transformed <- attr.transform.add(testt)
test.scaled <- predict(scaler, test.transformed)
preds <- predict(knn.model, test.scaled)
preds[outliers.test.por.la.cara] <- 0
sub.46 <- preds
createSubmission(sub.46, "46")


# ---------------------------------------------------------------- #
# SUBIDA 47
# 39 + k=21, kernel=epanechnikov
k <- 21
kernel <- "epanechnikov"
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
knn.model <- train.kknn(C ~ ., train.scaled.2, ks = k, kernel = kernel, scale=F)

testt <- data.frame(test)
testt[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
test.transformed <- attr.transform.add(testt)
test.scaled <- predict(scaler, test.transformed)
preds <- predict(knn.model, test.scaled)
preds[outliers.test.por.la.cara] <- 0
sub.47 <- preds
createSubmission(sub.47, "47")


# ---------------------------------------------------------------- #
# SUBIDA 50
# 39, aprendiendo distancias con la modificación de NCA
k <- 17
kernel <- "epanechnikov"
library(bmrm)
set.seed(28)
train.cleaned.2 <- CVCF(train, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)
X <- train.scaled.2[,-ncol(train.scaled.2)]
y <- train.scaled.2[, ncol(train.scaled.2)]
partitions <- balanced.cv.fold(train.scaled.2$C, num.cv = 9) # Particionamos los datos en nueve muestras igualmente distribuidas (si no, el algoritmo es demasiado lento).
train.predict.cnca <- function(i, train){
  partition.indexes <- which(partitions==i)
  Xtra <- train.scaled.2[partition.indexes, -ncol(train.scaled.2)]
  ytra <- train.scaled.2[partition.indexes, ncol(train.scaled.2)]
  
  # Aprendizaje de distancias
  cnca = dml$CNCA(cnn_thresh=0.5, remove_cnn_thresh=0.95, max_iter=25, eta0=0.01)
  cnca$fit(Xtra, ytra)
  X.transformed <- cnca$transform(X)
  train.dml <- data.frame(X.transformed, C = y)
  names(train.dml) <- names(train.transformed.2)
  # K-NN
  knn.model.2 <- train.kknn(C ~ ., train.dml, ks = k, kernel = kernel, scale=F)
  # Predicción
  testt <- data.frame(test)
  testt[outliers.test.por.la.cara,] <- 1 # Para que no estorben en las transformaciones
  test.transformed <- attr.transform.add(testt) 
  test.scaled <- predict(scaler, test.transformed)
  test.dml <- cnca$transform(test.scaled)
  test.dml <- as.data.frame(test.dml)

  names(test.dml) <- names(test.transformed)
  preds <- predict(knn.model.2, test.dml)
  preds[outliers.test.por.la.cara] <- 0
  preds
}

sub.50.all <- sapply(1:9, train.predict.cnca, train = train.scaled.2)
sub.50 <- apply(sub.50.all, 1, function(x) ifelse(sum(x==1) > 4,1,0))
createSubmission(sub.50, "50")
