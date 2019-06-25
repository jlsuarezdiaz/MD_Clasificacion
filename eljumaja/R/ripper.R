

source("preprocesado.r")

# LIBRERÍAS 

library(RWeka)
library(caret)
library(DMwR) 
library(bmrm)
library(OneR)

# LECTURA DATOS

train <- read.csv("train.csv", na.strings = c("?", "NA", "NR", "na", "NaN", "nan"))
train$C <- as.factor(train$C)
test <- read.csv("test.csv", na.strings = c("?", "NA", "NR", "na", "NaN", "nan"))

# FUNCIONES AUXILIARES

# Crea fichero con formato subida
createSubmission <- function(pred, filename){
  sub <- cbind(Id = 1:length(pred), Prediction = as.numeric(as.character(pred)))
  write.csv(sub, paste0("subs-ripper/",filename), row.names = F)
  sub
}


# Calcula el score de validación cruzada para el dataset
# funcion.train.predict: función(train, test) que entrena el clasificador con train y devuelve las predicciones sobre test
cross_validation <- function(dataset, funcion.train.predict, folds = 10){
  fold.indexes <- balanced.cv.fold(dataset$C)
  return(mean(sapply(1:folds, cross_validation_fold, fold.indexes, dataset, funcion.train.predict)))
}

cross_validation_fold <- function(fold, indexes, dataset, funcion.train.predict){
  test.inds <- which(indexes==fold)
  test <- na.omit(dataset[test.inds,])
  train <- dataset[-test.inds,]
  ypred <- funcion.train.predict(train, test[,-ncol(test)])
  mean(ypred==test$C)
}

# Resuelve problema desbalanceo
solveUnbalance <- function(data,type='ubOver'){
  n <- ncol(data)
  new.data <- ubBalance(data[,1:n-1], as.factor(data$C), type=type, positive=1)
  balanced <- cbind(new.data$X,C=new.data$Y)
  return (balanced)
}

# Comprueba si hay valores perdidos en cada fila
has.na <- function(x) apply(x,1,function(z)any(is.na(z)))


cbrt <- function(x) sign(x) * abs(x)^(1/3)
# Estudio de outliers, simetría y transformaciones a partir de los boxplots, uno a uno.
# X1 bien
# X2 le siente bien un logaritmo. posible outlier en X2 > 6 y clase 1
# X3 también le sienta bien un logaritmo.
# X4 bien, pero también le sienta bien logaritmo
# X5 bien
# X6 bien, X6 < 0 poner a 0
# X7 le sienta bien logaritmo si eliminamos los X7 = 0
# X8 tal vez X^2?
# X9 le sienta bien logaritmo, posible outlier en X9 > 400 y clase 1
# X10 bien
# X11 bien
# X12 bien
# X13 le sienta bien al cuadrado
# X14 bien
# X15 le sienta bien logaritmo
# X16 le sienta bien logaritmo
# X17 le sienta bien logaritmo
# X18 le sienta bien logaritmo
# X19 bien
# X20 bien
# X21 le sienta bien raiz cubica ?
# X22 bien
# X23 le sienta bien logaritmo + 100 (< 0 = 0)
# X24 le sienta bien logaritmo
# X25 le sienta bien cuadrado
# X26 le sienta bien raiz cubica
# X27 le sienta bien raíz cúbica
# X28 le sienta bien logaritmo
# X29 le sienta bien logaritmo
# X30 bien
# X31 le sienta bien logaritmo
# X32 bien
# X33 le sienta bien logaritmo
# X34 le sienta bien cuadrado
# X35 le sienta bien log(.+0.1), los no ceros = 0
# X36 bien
# X37 bien
# X38 bien
# X39 le sienta bien log
# X40 le sienta bien cbrt
# X41 bien
# X42 bien
# X43 le sienta bien cbrt
# X44 le sienta bien log
# X45 tal vez sqrt? y <0 a 0
# X46 bien
# X47 le sienta bien cbrt
# X48 log + 25
# X49 cbrt
# X50 bien

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
  # trans$X21 <- cbrt(trans$X21)
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
  ##
  
  trans
}


#summary(train)
#summary(test)


set.seed(123) # fijamos semilla

#---- SUBIDA 1 ----

funcion.train.predict.1 <- function(train, test){
  # Train
  model <- JRip(C ~ ., train)
  # Predict
  pred <- predict(model, test)
  return(pred)
}

#---- SUBIDA 2 ----

#Quitamos filas que toman el mismo valor -68000 y algo en muchas de sus variables -> sospechoso
#La mayoría de estas filas en train tienen etiqueta 0 -> asignamos 0 a filas en test con este valor

funcion.train.predict.2 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  # Train
  model <- JRip(C ~ ., train[-outliers.train,])
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 

#---- SUBIDA 3 ----

# Quitamos filas con NAs en train

funcion.train.predict.3 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  indices.nas.train <- which(has.na(train))
  
  # Train
  model <- JRip(C ~ ., train[-c(outliers.train, indices.nas.train),])
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 


#---- SUBIDA 4 ----

# Imputamos NAs en train

funcion.train.predict.4 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  if(length(outliers.train) > 0){
    train <- train[-outliers.train,]
  }
  # Imputación NAs por media
  imputed <- mice(train, m=1, method = "mean")
  train <- complete(imputed)
  
  
  # Train
  model <- JRip(C ~ ., train)
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 

#---- SUBIDA 5 ----

# Imputamos NAs en train con KNN

funcion.train.predict.5 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  if(length(outliers.train) > 0){
    train <- train[-outliers.train,]
  }
  # Imputación NAs con kNN
  train <- knnImputation(train) 
  
  # Train
  model <- JRip(C ~ ., train)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 


#---- SUBIDA 6 ----

# Subida 3 + quitamos atributos poco relacionados con la clase (<0.01)

# Quitamos columnas menos correladas con la clase
#train$C <- as.numeric(levels(train$C))[train$C]
#abs(cor(train[-indices.nas.train,])[,ncol(train)])<0.01

funcion.train.predict.6 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  indices.nas.train <- which(has.na(train))
  
  train <- train[,-c(41,44,31,48,1)]
  # Train
  model <- JRip(C ~ ., train[-c(outliers.train, indices.nas.train),])
  print(model)
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 



#--- SUBIDA 7 ----

# Subida 3 + quitamos atributos poco relacionados con la clase (<0.009)

funcion.train.predict.7 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  indices.nas.train <- which(has.na(train))
  
  train <- train[,-c(41,44,31,48)]
  # Train
  model <- JRip(C ~ ., train[-c(outliers.train, indices.nas.train),])
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 


#---- SUBIDA 8 ----

table(train$C)  # Desbalanceada

# Subida 3 + balanceo

funcion.train.predict.8 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  indices.nas.train <- which(has.na(train))
  
  train <- train[-c(outliers.train, indices.nas.train),]
  balanced <- solveUnbalance(train, type="ubUnder")
  
  # Train
  model <- JRip(C ~ ., balanced)
  #print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
}


#---- SUBIDA 9 ----
# Subida 2 + imputación knn por clases

funcion.train.predict.9 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  #indices.nas.train <- which(has.na(train))
  
  train <- train[-outliers.train,]
  
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Training
  model <- JRip(C ~ ., train.completed)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 


#---- SUBIDA 10 ----
# Subida 9 + EF por consenso

funcion.train.predict.10 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- EF(train.completed, consensus = T)$cleanData
  
  # Training
  model <- JRip(C ~ ., train.cleaned)
  print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 

#---- SUBIDA 12 ----
# Subida 9 + CVCF por consenso

funcion.train.predict.12 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- CVCF(train.completed, consensus = T)$cleanData
  
  # Training
  model <- JRip(C ~ ., train.cleaned)
  print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 


#---- SUBIDA 13 ----

# Subida 9 + CVCF por mayoría

funcion.train.predict.13 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  
  # Training
  model <- JRip(C ~ ., train.cleaned)
  print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
}


#---- SUBIDA 14 ----
# Subida 9 + IPF por consenso

funcion.train.predict.14 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- IPF(train.completed, consensus = T)$cleanData
  
  # Training
  model <- JRip(C ~ ., train.cleaned)
  print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 

#---- SUBIDA 15 ----

# Subida 9 + IPF por mayoría

funcion.train.predict.15 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- IPF(train.completed, consensus = F)$cleanData
  
  # Training
  model <- JRip(C ~ ., train.cleaned)
  print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
} 





#---- SUBIDA 16 ----
# Subida 9 + EF por mayoría

funcion.train.predict.16 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- EF(train.completed, consensus = F)$cleanData
  
  # Training
  model <- JRip(C ~ ., train.cleaned)
  print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
}


#---- SUBIDA 17 ----
# Subida 16 + transformaciones

funcion.train.predict.17 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- EF(train.completed, consensus = F)$cleanData
  
  # Transformamos atributos
  train.transformed <- attr.transform.add(train.cleaned)
  
  # Training
  model <- JRip(C ~ ., train.transformed)
  print(model)
  
  # Predict
  test[outliers.test,] <- 1
  test.transformed <- attr.transform.add(test)
  pred <- predict(model, test.transformed)
  pred[outliers.test] <- 0
  return(pred)
}


#---- SUBIDA 18 ----
# Subida 16 + transformaciones con escalado

funcion.train.predict.18 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  # Limpiamos ruido
  train.cleaned <- EF(train.completed, consensus = F)$cleanData
  
  # Transformamos atributos
  train.transformed <- attr.transform.add(train.cleaned)
  scaler <- preProcess(train.transformed) # Centrado y escalado
  train.scaled <- predict(scaler, train.transformed)
  
  # Training
  model <- JRip(C ~ ., train.scaled)
  print(model)
  
  # Predict
  test[outliers.test,] <- 1
  test.transformed <- attr.transform.add(test)
  test.scaled <- predict(scaler, test.transformed)
  pred <- predict(model, test.scaled)
  pred[outliers.test] <- 0
  return(pred)
}


#---- SUBIDA 19 ----
# Primero filtro ruido CVCF por mayoría y después KNN imputación por clases

funcion.train.predict.19 <- function(train, test){
  # Preprocesamiento
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  train <- train[-outliers.train,]
  
  # Limpiamos ruido
  train.cleaned <- CVCF(train, consensus = F)$cleanData
  
  # Imputamos NAs con KNN por clases
  train.completed.0 <- knnImputation(train.cleaned[train.cleaned$C == 0,]) 
  train.completed.1 <- knnImputation(train.cleaned[train.cleaned$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  
  # Training
  model <- JRip(C ~ ., train.completed)
  print(model)
  
  # Predict
  pred <- predict(model, test)
  pred[outliers.test] <- 0
  return(pred)
}

sub <- funcion.train.predict.16(train, test)
createSubmission(sub, "test")












