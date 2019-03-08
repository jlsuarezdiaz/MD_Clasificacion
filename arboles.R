# lectura de datos
library(ggplot2)
library(caret)
library(RKEEL)
# library(rDML) # Por si acaso
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

library("tree")
library(rpart)
require(discretization)
library(party)
library(RWeka)
library(Amelia)

train <- read.csv("./train.csv", header=TRUE, na.strings="?")
test <- read.csv("./test.csv", header = TRUE, na.strings = "?")

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

cbrt <- function(x) sign(x) * abs(x)^(1/3)


# 1-2

# preprocesamiento
# aprendizaje del modelo

model = rpart(as.factor(train$C)~.,data = train, method = "class",
              control = rpart.control(minsplit = 10, xval = 10))

# prediccion de etiquetas
model.pred.train = as.numeric(predict(model, train, type = "class")) 
model.pred.train[which(model.pred.train == 1)] = 0
model.pred.train[which(model.pred.train == 2)] = 1

round(mean(model.pred.train==train$C)*100,digits=1) 

model.pred = as.numeric(predict(model, test, type = "class"))
model.pred

model.pred[which(model.pred == 1)] = 0
model.pred[which(model.pred == 2)] = 1
model.pred
# 1 -> 2
# 0 -> 1
write.csv(data.frame('Id'=c(1:length(model.pred)),'Prediction'=model.pred), file = paste("RAW.csv"), row.names=FALSE)


# 3

# preprocesamiento

train = computeMissingValues(train, type = "knn", k = 2)
new_data = preProcessData(train,test)

scaled.train = new_data[[1]]
scaled.test = new_data[[2]]

# aprendizaje del modelo
model = rpart(as.factor(scaled.train$C)~.,data = scaled.train, method = "class",
              control = rpart.control(minsplit = 10, xval = 10))

# prediccion de etiquetas
model.pred.train = as.numeric(predict(model, scaled.train, type = "class")) 
model.pred.train[which(model.pred.train == 1)] = 0
model.pred.train[which(model.pred.train == 2)] = 1

round(mean(model.pred.train==train$C)*100,digits=1) 


model.pred = as.numeric(predict(model, scaled.test, type = "class"))
model.pred

model.pred[which(model.pred == 1)] = 0
model.pred[which(model.pred == 2)] = 1
model.pred
# 1 -> 2
# 0 -> 1
write.csv(data.frame('Id'=c(1:length(model.pred)),'Prediction'=model.pred), file = paste("2KNNCS.csv"), row.names=FALSE)


# 4

# preprocesamiento

train = computeMissingValues(train, type = "median")
new_data = preProcessData(train,test)

scaled.train = new_data[[1]]
scaled.test = new_data[[2]]

# aprendizaje del modelo
model = rpart(as.factor(scaled.train$C)~.,data = scaled.train, method = "class",
              control = rpart.control(minsplit = 10, xval = 10))
model = prune(model, cp = model$cptable[which.min(model$cptable[,"xerror"]),"CP"])

# prediccion de etiquetas
model.pred.train = as.numeric(predict(model, scaled.train, type = "class")) 
model.pred.train[which(model.pred.train == 1)] = 0
model.pred.train[which(model.pred.train == 2)] = 1

round(mean(model.pred.train==train$C)*100,digits=1) 


model.pred = as.numeric(predict(model, scaled.test, type = "class"))

model.pred[which(model.pred == 1)] = 0
model.pred[which(model.pred == 2)] = 1

# 1 -> 2
# 0 -> 1
write.csv(data.frame('Id'=c(1:length(model.pred)),'Prediction'=model.pred), file = paste("MedianCS.csv"), row.names=FALSE)


# 5


# preprocesamiento
ini = mice(train, maxit = 0)
quitar = as.character(ini$loggedEvents[,"out"])
valores = mice(train, meth="pmm", seed = 500, remove_collinear = FALSE)
compData = complete(valores,1)
train_1 = compData

new_data = preProcessData(train_1,test)

scaled.train = new_data[[1]]
scaled.test = new_data[[2]]

# aprendizaje del modelo
model = rpart(as.factor(scaled.train$C)~.,data = scaled.train, method = "class",
              control = rpart.control(minsplit = 10, xval = 10))

# prediccion de etiquetas
model.pred.train = as.numeric(predict(model, scaled.train, type = "class")) 
model.pred.train[which(model.pred.train == 1)] = 0
model.pred.train[which(model.pred.train == 2)] = 1

round(mean(model.pred.train==train$C)*100,digits=1) 


model.pred = as.numeric(predict(model, scaled.test, type = "class"))
model.pred

model.pred[which(model.pred == 1)] = 0
model.pred[which(model.pred == 2)] = 1
model.pred
# 1 -> 2
# 0 -> 1
write.csv(data.frame('Id'=c(1:length(model.pred)),'Prediction'=model.pred), file = paste("MedianCS.csv"), row.names=FALSE)


# 8

# preprocesamiento
data.sin.na = computeMissingValues(train, type = 'rf')
new_data = preProcessData(data.sin.na,test)

scaled.train = new_data[[1]]
scaled.test = new_data[[2]]

# aprendizaje del modelo
model = rpart(as.factor(scaled.train$C)~.,data = scaled.train, method = "class",
              control = rpart.control(minsplit = 10, xval = 10))

model = prune(model,cp = model$cptable[which.min(model$cptable[,"xerror"]),"CP"])

# prediccion de etiquetas
model.pred.train = as.numeric(predict(model, scaled.train, type = "class")) 
model.pred.train[which(model.pred.train == 1)] = 0
model.pred.train[which(model.pred.train == 2)] = 1

round(mean(model.pred.train==scaled.train$C)*100,digits=1) 


model.pred = as.numeric(predict(model, scaled.test, type = "class"))
model.pred

model.pred[which(model.pred == 1)] = 0
model.pred[which(model.pred == 2)] = 1
model.pred
# 1 -> 2
# 0 -> 1
write.csv(data.frame('Id'=c(1:length(model.pred)),'Prediction'=model.pred), file = paste("RFCS.csv"), row.names=FALSE)



# 9-11

library(FSelector)

Class = train[,51]
train_1 = train[,-51]

weights = FSelector::linear.correlation(Class~.,data = train_1)
subset = FSelector::cutoff.k(weights,30)
f1 = as.simple.formula(subset,"Class")

subset = FSelector::cutoff.k(weights,3)
f2 = as.simple.formula(subset,"Class")

weights = FSelector::rank.correlation(Class~.,data = train_1)
subset = FSelector::cutoff.k(weights,7)
f3 = as.simple.formula(subset,"Class")

new_data = preProcessData(cbind(train_1,Class),test)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]

# aprendizaje del modelo
model = rpart(f3,data = scaled.train[,subset], method = "class",
              control = rpart.control(minsplit = 10, xval = 10))

model = prune(model,cp = model$cptable[which.min(model$cptable[,"xerror"]),"CP"])

# prediccion de etiquetas
model.pred.train = as.numeric(predict(model, scaled.train, type = "class")) 
model.pred.train[which(model.pred.train == 1)] = 0
model.pred.train[which(model.pred.train == 2)] = 1


round(mean(model.pred.train==scaled.train$C)*100,digits=1) 


model.pred = as.numeric(predict(model, scaled.test, type = "class"))
model.pred

model.pred[which(model.pred == 1)] = 0
model.pred[which(model.pred == 2)] = 1
model.pred
# 1 -> 2
# 0 -> 1
write.csv(data.frame('Id'=c(1:length(model.pred)),'Prediction'=model.pred), file = paste("RC7MEDCS.csv"), row.names=FALSE)


# 12-14

library(NoiseFiltersR)
library(FSelector)


train_1 = computeMissingValues(train, type = "knn", k = 1)
train_1[,51] <- as.factor(train_1[,51])
set.seed(1)
out.data <- NoiseFiltersR::IPF(train_1, nfolds = 5, consensus = FALSE)
data.clean = out.data$cleanData
new_data = preProcessData(data.clean,test)

scaled.train = new_data[[1]]
scaled.test = new_data[[2]]


Class = scaled.train[,51]
weights <- FSelector::random.forest.importance(Class ~ .,scaled.train[,-51], importance.type = 1)

print(weights)

subset <- cutoff.k(weights,10) 
subset1 <- cutoff.k(weights,20) 

subset

f <- as.simple.formula(subset,"C")
f1 <- as.simple.formula(subset,"C")
  
control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 5)
modelo <- caret::train(f, data = scaled.train, method="ctree", trControl = control)

confusionMatrix(modelo)

pred.train = predict(modelo,scaled.train, type = "prob")
fit.train = ifelse(pred.train[1]> 0.5,0,1)

round(mean(fit.train==scaled.train$C)*100,digits=1) 

pred.test = predict(modelo,scaled.test, type = "prob")
pred.test
fit.test = ifelse(pred.test[1]> 0.5,0,1)

fit.test
write.csv(data.frame('Id'=c(1:length(fit.test)),'Prediction'=fit.test), file = paste("IPFRFICTREE.csv"), row.names=FALSE)
#write.csv(data.frame('Id'=c(1:length(fit.test)),'Prediction'=fit.test), file = paste("IPFRFICTREE20.csv"), row.names=FALSE)


# 15 

# preprocesamiento

train.sin.na = computeMissingValues(train, type = 'rf')
train.sin.na[,51] = as.factor(train.sin.na[,51])
train.sin.ruido = filterNoiseData(train.sin.na)

train.sin.ruido[,51] = as.numeric(train.sin.ruido[,51])
train.sin = computeOutliers(train.sin.ruido, type = 'remove')
new_data = preProcessData(train.sin,test)

scaled_train = new_data[[1]]
scaled_test = new_data[[2]]

# comentar  para quitar la discretización.
cm <- discretization::disc.Topdown(scaled_train,1)
scaled_train = cm$Disc.data

Class = scaled_train[,51]
predictores.1 = rankingLearningRandomForest(scaled_train[,-51],Class,numeroVars = c(0,1))
p1 = predictores.1[1:10]

predictores.1 = featureSelection('chi',10, scaled_train[,-51],Class)
p1=predictores.1

#Class = scaled_train[,51]
#predictores.2 = featureSelection('rfi',10,scaled_train[,-51],Class)
#predictores.2


f1 = as.simple.formula(p1,"C")
#f2 = as.simple.formula(predictores.2,"C")

# convertimos en facctor
tmp = as.numeric(scaled_train[,51])

tmp[which(tmp == 1)] = 0
tmp[which(tmp == 2)] = 1

scaled_train[,51] = as.factor(tmp)

control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 5)
modelo <- caret::train(f1, data = scaled_train, method="ctree", trControl = control)

confusionMatrix(modelo)

pred.train = predict(modelo,scaled_train, type = "prob")
fit.train = ifelse(pred.train[1]> 0.5,0,1)

round(mean(fit.train==scaled_train$C)*100,digits=1) 

pred.test = predict(modelo,scaled_test, type = "prob")
fit.test = ifelse(pred.test[1]> 0.5,0,1)

# estos son los trialNumero.csv
write.csv(data.frame('Id'=c(1:length(fit.test)),'Prediction'=fit.test), file = paste("trial3.csv"), row.names=FALSE)






######

## cambiar knn por median
train.sin.na = computeMissingValues(train, type = "knn")
train.sin.outliers = computeOutliers(train.sin.na, type = 'mean')
train.sin.ruido = filterNoiseData(train.sin.outliers)

new_data = preProcessData(train.sin.ruido,test)

scaled_train = new_data[[1]]
scaled_test = new_data[[2]]

scaled_train[,51] =  scaled_train[,51]-1
train_1 = solveUnbalance(scaled_train)

predictores.1 = featureSelection('rfi',25, train[,-51],train[,51])
p1=predictores.1
p1
f1 = as.simple.formula(p1,"Y")


model = ctree(f1,data = train_1,control = ctree_control(mincriterion = 0.9))

# prediccion de etiquetas
model = prune(model,cp = 0.01)
model =prune.misclass (model ,best =3)

model.pred.train = predict(model, train, type = "prob")
fit.train = ifelse(model.pred.train[[1]]> 0.5,0,1)

model.pred.train

model.pred.train
model.pred.train[which(model.pred.train == 1)] = 0
model.pred.train[which(model.pred.train == 2)] = 1

round(mean(model.pred.train==train$C)*100,digits=1) 


model.pred = as.numeric(predict(model, scaled_test, type = "class"))
model.pred

model.pred[which(model.pred == 1)] = 0
model.pred[which(model.pred == 2)] = 1
model.pred



### PCA + unbalance

datos.pca = prcomp(train_1,scale = TRUE)
library(factoextra)
std_dev <- datos.pca$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
plot(cumsum(prop_varex), xlab = "PCA", type = "b")
data.train = data.frame(datos.pca$x,Y)
test.data = predict(datos.pca, newdata = scaled_test)
test.data = as.data.frame(test.data)

control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 5)
modelo <- caret::train(Y~., data = data.train[,1:30], method="ctree", trControl = control)

confusionMatrix(modelo)

pred.train = predict(modelo,data.train, type = "prob")
fit.train = ifelse(pred.train[1]> 0.5,0,1)
pred.train
round(mean(fit.train==data.train$Y)*100,digits=1) 

pred.test = predict(modelo,test.data, type = "prob")
pred.test
fit.test = ifelse(pred.test[1]> 0.5,0,1)
head(pred.test[1])
head(fit.test)


write.csv(data.frame('Id'=c(1:length(fit.test)),'Prediction'=fit.test), file = paste("fsrpart.csv"), row.names=FALSE)




## amelia y unbalance correlation

train.sin.na = computeMissingValues(train,type = 'mean')

train.sin.na = removeHighCorrelationAttributes(train.sin.na,0.99)



###################################################
missmap(train)


completed_data <- amelia(train, m = 1,p2s = 2)

train1 = completed_data$imputations[[1]]
train2 = completed_data$imputations[[2]]
train3 = completed_data$imputations[[3]]
train4 = completed_data$imputations[[4]]
train5 = completed_data$imputations[[5]]
#######################################################


# cambiar train.sin.na por trainNum que son los conjuntos de amelia.
train.sin.outliers = computeOutliers(train1, type = 'median')
train.sin.ruido = filterNoiseData(train.sin.outliers)

train.sin.ruido = solveUnbalance(train.sin.ruido)

new_data = preProcessData(train.sin.ruido,test)

scaled_train = new_data[[1]]
scaled_test = new_data[[2]]


p1 = featureSelection('rfi',10,train[,-ncol(scaled_train)], as.factor(train[,ncol(scaled_train)]))
f = as.simple.formula(p1,"C")


control <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 5)
modelo <- caret::train(f, data = train, method="ctree", trControl = control)

confusionMatrix(modelo)

pred.train = predict(modelo,train, type = "prob")
fit.train = ifelse(pred.train[1]> 0.5,0,1)

round(mean(fit.train==train$C)*100,digits=1) 

pred.test = predict(modelo,test, type = "prob")
pred.test
fit.test = ifelse(pred.test[1]> 0.5,0,1)

fit.test
write.csv(data.frame('Id'=c(1:length(fit.test)),'Prediction'=fit.test), file = paste("unbalancecorrelation.csv"), row.names=FALSE)



######### nuevas pruebas #########

library(bmrm)

# Calcula el score de validación cruzada para el dataset
# funcion.train.predict: función(train, test) que entrena el clasificador con train y devuelve las predicciones sobre test
cross_validation <- function(dataset, funcion.train.predict, folds = 10){
  fold.indexes <- balanced.cv.fold(dataset$C)
  return(mean(sapply(1:folds, cross_validation_fold, fold.indexes, dataset, funcion.train.predict)))
}

cross_validation_fold <- function(fold, indexes, dataset, funcion.train.predict){
  train.inds <- which(indexes==fold)
  train <- dataset[train.inds,]
  test <- na.omit(dataset[-train.inds,])
  ypred <- funcion.train.predict(train, test[,-ncol(test)])
  mean(ypred==test$C)
}

set.seed(28)

funcion.train.predict <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  indices.nas.train <- which(has.na(train))
  model <- caret::train(as.factor(C) ~ ., train[-c(outliers.train.por.la.cara, indices.nas.train),], method="ctree", preProcess = c("center", "scale"))
  
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  print(preds)
  return(preds)
}

has.na <- function(x) apply(x,1,function(z)any(is.na(z)))

createSubmission <- function(pred, filename){
  sub <- cbind(Id = 1:length(pred), Prediction = as.numeric(as.character(pred)))
  write.csv(sub, paste0("subs-tree/",filename), row.names = F)
  sub
}

cross_validation(train, funcion.train.predict)

sub.prueba <- funcion.train.predict(train, test)
sub.prueba
sub <- createSubmission(sub.prueba, "pruebaseguronomejora") # 0.83 ??????????


### prueba numero 32

set.seed(28)

train.predict.32 <- function(train, test){
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
  model <- caret::train(C ~ ., train.cleaned, method="ctree", preProcess = c("YeoJohnson","center", "scale"), 
                        tuneGrid = expand.grid(mincriterion = c(0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90)))

  #print(model)
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.34)

sub.34 <- train.predict.32(train, test)
createSubmission(sub.32, "34")

## prueba 33 # -> 0.882

set.seed(28)
train.predict.knn.imputation.ef <- function(train, test){
  # Train
  
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- EF(train.completed)$cleanData
  
  model <- caret::train(C ~ ., train.cleaned, method="ctree", preProcess = c("YeoJohnson","center", "scale"), 
                        tuneGrid = expand.grid(mincriterion = c(0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90)))
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.imputation.ef)

set.seed(28)

sub.33 <- train.predict.knn.imputation.ef(train, test)
createSubmission(sub.33, "33")


#### prueba 34

train.predict.34 <- function(train, test){
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
  Class = train.cleaned[,ncol(train.cleaned)]
  weights <- FSelector::random.forest.importance(Class ~ .,train.cleaned[,-ncol(train.cleaned)], importance.type = 1)
  subset <- cutoff.k(weights,10) 
  f = as.simple.formula(subset, "C")  
  model <- caret::train(f, train.cleaned, method="ctree", preProcess = c("YeoJohnson","center", "scale"), 
                        tuneGrid = expand.grid(mincriterion = c(0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90)))
  
  #print(model)
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.34)

set.seed(28)
sub.34 <- train.predict.34(train, test)
createSubmission(sub.34, "34")

#### prueba 35

set.seed(28)
train.predict.knn.imputation.ef.majority <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- EF(train.completed, consensus = F)$cleanData
  
  model <- caret::train(C ~ ., train.cleaned, method="ctree", preProcess = c("YeoJohnson","center", "scale"), 
                        tuneGrid = expand.grid(mincriterion = c(0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90)))
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.imputation.ef.majority)

set.seed(28)
sub.35 <- train.predict.knn.imputation.ef.majority(train, test)
createSubmission(sub.35, "35")

### prueba 36

set.seed(28)
train.predict.knn.imputation.cvcf <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = T)$cleanData
  
  model <- caret::train(C ~ ., train.cleaned, method="ctree", preProcess = c("center", "scale"), 
                        tuneGrid = expand.grid(mincriterion = c(0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90)))
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.imputation.cvcf)

set.seed(28)
sub.36 <- train.predict.knn.imputation.cvcf(train, test)
createSubmission(sub.36, "36")



##### prueba 37

set.seed(28)
train.predict.37 <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- EF(train.completed)$cleanData
  train.transformed <- attr.transform.add(train.cleaned)
  scaler <- preProcess(train.transformed) # Centrado y escalado
  train.scaled <- predict(scaler, train.transformed)
  
  
  model = ctree(C ~ .,data = train.scaled,control = ctree_control(mincriterion = 0.9))
  
  # Predict
  test[outliers.test.por.la.cara,] <- 1
  test.transformed <- attr.transform.add(test)
  test.scaled <- predict(scaler, test.transformed)
  preds <- predict(model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

set.seed(28)
train$C <- as.factor(train$C)

sub.37 <- train.predict.37(train, test)
createSubmission(sub.37, "37") 


##### prueba 38

set.seed(28)
train.predict.knn.imputation.ipf <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- IPF(train.completed, consensus = T)$cleanData
  
  model = rpart(C ~.,data = train.cleaned, method = "class",
                control = rpart.control(minsplit = 10, xval = 10))
  model = prune(model, cp = model$cptable[which.min(model$cptable[,"xerror"]),"CP"])
  
  # Predict
  preds <- predict(model, test, type="class")
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.imputation.ipf)

set.seed(28)
sub.38 <- train.predict.knn.imputation.ipf(train, test)
createSubmission(sub.38, "38")


### prueba 39

set.seed(28)
train.predict.knn.imputation.ef.transforms <- function(train, test){
  # Train
  
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- EF(train.completed)$cleanData
  train.transformed <- attr.transform.add(train.cleaned)
  scaler <- preProcess(train.transformed) # Centrado y escalado
  train.scaled <- predict(scaler, train.transformed)
  
  model <- caret::train(C ~ ., train.scaled, method="ctree", preProcess = c("YeoJohnson"), 
                        tuneGrid = expand.grid(mincriterion = c(0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90)))

  
  # Predict
  
  test[outliers.test.por.la.cara,] <- 1
  test.transformed <- attr.transform.add(test)
  test.scaled <- predict(scaler, test.transformed)
  preds <- predict(model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}



train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.imputation.ef.transforms)

set.seed(28)

sub.39 <- train.predict.knn.imputation.ef.transforms(train, test)
createSubmission(sub.39, "39")

### prueba 40

set.seed(28)
train.predict.40<- function(train, test){
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
  train.cleaned <- EF(train.completed)$cleanData

  model <- caret::train(C ~ ., train.cleaned, method="ctree", preProcess = c("YeoJohnson","center", "scale"), 
                        tuneGrid = expand.grid(mincriterion = c(0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90)))
  
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.40)

set.seed(28)

sub.40 <- train.predict.40(train, test)
createSubmission(sub.40, "40")



### prueba 41

train.predict.41 <- function(train,test){
  # Train
  
  outliers.train <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  #indices.nas.train <- which(has.na(train))
  
  train <- train[-outliers.train,]
  
  train.completed.0 <- knnImputation(train[train$C == 0,]) 
  train.completed.1 <- knnImputation(train[train$C == 1,])
  train.completed <- rbind(train.completed.0, train.completed.1)
  
  model = ctree(C ~ .,data = train.completed,control = ctree_control(mincriterion = 0.9))  # Predict
  
  preds <- predict(model, test)
  preds[outliers.test] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.41)

set.seed(28)
sub.41 <- train.predict.41(train, test)
createSubmission(sub.41, "41")


### prueba 42

set.seed(28)
train.predict.42<- function(train, test){
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
  train.cleaned <- EF(train.completed)$cleanData
  
  model = ctree(C ~ .,data = train.completed)  
  
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.42)

set.seed(28)

sub.42 <- train.predict.42(train, test)
createSubmission(sub.42, "42")


### prueba 43
require(imbalance)

train.predict.43 <- function(train, test){
  # Train
  
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- amelia(train, m = 1,p2s = 2)$imputations[[1]]

  train.completed$C <- as.factor(train.completed$C)
  train.cleaned <- EF(as.data.frame(train.completed))$cleanData

  ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                       sampling = "smote")
  
  model <- caret::train(C ~ ., train.cleaned, method="ctree", preProcess = c("YeoJohnson","center", "scale"), trControl = ctrl,
                        tuneGrid = expand.grid(mincriterion = c(0.95,0.94,0.93,0.92,0.91,0.90)))
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}


cross_validation(train, train.predict.43)

set.seed(28)
sub.43 = preds
sub.43 <- train.predict.43(train, test)
createSubmission(sub.43, "43")


### prueba 44

set.seed(28)
train.predict.knn.44 <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- EF(train.completed)$cleanData
  
  model <- caret::train(C ~ ., train.cleaned, method="J48", preProcess = c("center", "scale"), 
                        tuneGrid = expand.grid(C = c(0.1,0.15,0.2,0.25), M  = c(1,2,3,4,5)))
  
  model$bestTune
  # entrenamos con el mejor C,M
  model = J48(C~., data=train.cleaned, control = Weka_control(M = 4, C = 0.1 ))
  
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.44)

set.seed(28)
sub.44 = preds
sub.44 <- train.predict.knn.44(train, test)
createSubmission(sub.44, "44")


### prueba 45

train.predict.knn.45 <- function(train, test){
  # Train
  outliers.train.por.la.cara <- which(apply(train[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  
  if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
  train.completed <- knnImputation(train) 
  train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
  
  Class = train.cleaned[,ncol(train.cleaned)]
  atributos = FSelector::random.forest.importance(Class~., train.cleaned[,-ncol(train.cleaned)])
  print(atributos)
  subset = FSelector::cutoff.k(atributos,25)
  f = as.simple.formula(subset,"C")
  
  model <- caret::train(f, train.cleaned, method="J48", preProcess = c("center", "scale"), 
                        tuneGrid = expand.grid(C = c(0.1,0.15,0.2,0.25), M  = c(1,2,3,4,5)))
  
  model$bestTune
  # entrenamos con el mejor C,M
  model = J48(C~., data=train, control = Weka_control(M = 4, C = 0.1))
  
  # Predict
  preds <- predict(model, test)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.45)

set.seed(28)

sub.45 <- train.predict.knn.45(train, test)
createSubmission(sub.45, "45")




## prueba 46

train.predict.46 <- function(train, test){
  # Train
    train.cleaned <- CVCF(train, consensus = F)$cleanData
    outliers.train.por.la.cara <- which(apply(train.cleaned[,-ncol(train.cleaned)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
    outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
    if(length(outliers.train.por.la.cara) > 0) train.cleaned <- train.cleaned[-outliers.train.por.la.cara,]
    
    
    train.transformed <- attr.transform.add(train.cleaned) %>% dplyr::select(-C,C)
    train.completed <- knnImputation(train.transformed) 
    scaler <- preProcess(train.completed)
    train.scaled <- predict(scaler, train.completed)
    
    model <- caret::train(C ~ ., train.scaled, method="J48", 
                          tuneGrid = expand.grid(C = c(0.1,0.15,0.2,0.25), M  = c(1,2,3,4,5)))
  
    model$bestTune
    # completar con el que salga el mejor
  model = J48(C~., data=train.scaled, control = Weka_control(M = 5, C = 0.1))
  
  # Predict
  test[outliers.test.por.la.cara,] <- 1
  test.transformed <- attr.transform.add(test)
  
  test.scaled <- predict(scaler, test.transformed)
  preds <- predict(model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}


train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.46)

set.seed(28)
sub.46 = preds
sub.46 <- train.predict.46(train, test)
createSubmission(sub.46, "46")


### prueba 47

train.predict.47 <- function(train, test){
  # Train
  train.cleaned <- CVCF(train, consensus = F)$cleanData
  outliers.train.por.la.cara <- which(apply(train.cleaned[,-ncol(train.cleaned)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train.cleaned <- train.cleaned[-outliers.train.por.la.cara,]
  
  
  train.transformed <- attr.transform.add(train.cleaned) %>% dplyr::select(-C,C)
  train.completed <- knnImputation(train.transformed) 
  scaler <- preProcess(train.completed)
  train.scaled <- predict(scaler, train.completed)
  
  model <- caret::train(C ~ ., train.scaled, method="ctree", 
                        tuneGrid = expand.grid(mincriterion = c(0.95,0.94,0.93,0.92,0.91,0.90)))
  
  model$bestTune
  # completar con el que salga el mejor
  model = ctree(C ~ .,data = train.scaled,control = ctree_control(mincriterion = 0.95))
  
  # Predict
  test[outliers.test.por.la.cara,] <- 1
  test.transformed <- attr.transform.add(test)
  
  test.scaled <- predict(scaler, test.transformed)
  preds <- predict(model, test.scaled)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}


train$C <- as.factor(train$C)
cross_validation(train, train.predict.knn.46)

set.seed(28)
sub.46 = preds
sub.46 <- train.predict.46(train, test)
createSubmission(sub.46, "46")


### prueba 48

train.predict.48 <- function(train, test){
  # Train
  train.cleaned <- CVCF(train, consensus = F)$cleanData
  outliers.train.por.la.cara <- which(apply(train.cleaned[,-ncol(train.cleaned)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
  if(length(outliers.train.por.la.cara) > 0) train.cleaned <- train.cleaned[-outliers.train.por.la.cara,]
  
  train.transformed <- attr.transform.add(train.cleaned) %>% dplyr::select(-C,C)
  train.completed <- knnImputation(train.transformed) 
  #tmp <- cor(train.completed[,-ncol(train.completed)])
  #tmp[upper.tri(tmp)] <- 0
  #diag(tmp) <- 0
  
  train.discretizado <- arulesCBA::discretizeDF.supervised(C~ .,train.completed, method= "mdlp")
  
  Class = train.discretizado[,ncol(train.discretizado)]
  pesos <- FSelector::chi.squared(Class ~., train.discretizado[,-ncol(train.discretizado)])
  variables <- FSelector::cutoff.k(pesos,25)

  f <- as.simple.formula(variables,"C")
  

  model <- caret::train(C ~ ., train.scaled, method="J48", 
                        tuneGrid = expand.grid(C = c(0.1,0.15,0.2,0.25), M  = c(1,2,3,4,5)))

  model <- caret::train(C ~ ., train.scaled, method="ctree", 
                        tuneGrid = expand.grid(mincriterion = c(0.95,0.94,0.93,0.92,0.91,0.90)))
  
  # completar con el que salga el mejor
  model = J48(f, data=train.discretizado, control = Weka_control(M = 5, C = 0.1))
  model = ctree(f,data = train.discretizado, control = ctree_control(mincriterion = 0.95))
  
  # Predict
  
  test[outliers.test.por.la.cara,] <- 1
  test.transformed <- attr.transform.add(test)
  
  test.discretized <- arules::discretizeDF(
    test.transformed,train.discretizado[,-ncol(train.discretizado)])
  
  preds <- predict(model, test.discretized)
  preds[outliers.test.por.la.cara] <- 0
  return(preds)
}

train$C <- as.factor(train$C)
set.seed(15)

# las ultimas subidas salen de aqui modificando la función y ejecutando paso a paso
sub.48 <- train.predict.48(train, test)
createSubmission(sub.48, "52")
