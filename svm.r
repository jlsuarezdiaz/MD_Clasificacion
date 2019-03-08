source("cv.R")
source("preprocesado.R")
require('e1071')

predictAndSave <- function(train, test,filename,kernel="radial"){
  model <- svm(C~.,data = train,kernel=kernel)
  
  pred <- predict(model,train)
  print(mean(round(pred)==train$C))
  
  pred <- predict(model,test)
  outliers.test <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
  pred[outliers.test] <- 0
  write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=round(pred)), file = filename, row.names=FALSE)
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



data <- read.csv("train.csv", header=TRUE, na.strings="?")
test.data <- read.csv("test.csv", header=TRUE, na.strings="?")


# 1
print(1)
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train = computeMissingValues(scaled.train, type = "remove")
predictAndSave(scaled.train,scaled.test,'subs-svm/1.csv')


# 2
print(2)
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "mean")
predictAndSave(scaled.train.knn,scaled.test,'subs-svm/2.csv')


# 3
print(3)
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "mean")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='remove')
scaled.train.outliers.knn$C <- scaled.train.knn$C
predictAndSave(scaled.train.outliers.knn,scaled.test,'subs-svm/3.csv')


# 4
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "mean")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='knn')
scaled.train.outliers.knn$C <- scaled.train.knn$C
predictAndSave(scaled.train.outliers.knn,scaled.test,'subs-svm/3.csv')


# 5
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
predictAndSave(scaled.train.knn,scaled.test,'subs-svm/5.csv')



# 6
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='knn')
scaled.train.outliers.knn$C <- scaled.train.knn$C
predictAndSave(scaled.train.outliers.knn,scaled.test,'subs-svm/6.csv')



# 7
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='mean')
scaled.train.outliers.knn$C <- scaled.train.knn$C
predictAndSave(scaled.train.outliers.knn,scaled.test,'subs-svm/7.csv')



# 8
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='mean')
scaled.train.outliers.knn$C <- scaled.train.knn$C
scaled.train.remove.low.cor <- removeLowCorrelationAttributes(scaled.train.outliers.knn)
predictAndSave(scaled.train.remove.low.cor,scaled.test,'subs-svm/8.csv')


# 9
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='mean')
scaled.train.outliers$C <- scaled.train.knn$C
scaled.train.outliers.balanced <- solveUnbalance(scaled.train.outliers, "ubOver")
predictAndSave(scaled.train.outliers.balanced,scaled.test,'subs-svm/9.csv')


# 10
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='mean')
scaled.train.outliers.knn$C <- scaled.train.knn$C
scaled.train.outliers.knn.balanced <- solveUnbalance(scaled.train.outliers.knn, "ubUndersampling")
predictAndSave(scaled.train.outliers.knn.balanced,scaled.test,'subs-svm/10.csv')



# 11
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='mean')
scaled.train.outliers.knn$C <- scaled.train.knn$C
scaled.train.outliers.balanced <- solveUnbalance(scaled.train.outliers, "ubSMOTE")
predictAndSave(scaled.train.outliers.balanced,scaled.test,'subs-svm/11.csv')



# 12
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='rf')
scaled.train.outliers.knn$C <- scaled.train.knn$C
predictAndSave(scaled.train.outliers.knn,scaled.test,'subs-svm/12.csv')




# 13
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='rf')
scaled.train.outliers$C <- scaled.train.knn$C
scaled.train.outliers.rf.balanced <- solveUnbalance(scaled.train.outliers, "ubOver")
predictAndSave(scaled.train.outliers.rf.balanced,scaled.test,'subs-svm/13.csv')




# 14
outliers.train <- which(apply(data[,-ncol(data)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
outliers.test <- which(apply(test.data, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
new_data = preProcessData(data[-outliers.train,],test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
predictAndSave(scaled.train.knn, scaled.test, 'missing-knn-sin-outliers-raros.csv', kernel="radial")



# 15
outliers.train <- which(apply(data[,-ncol(data)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
outliers.test <- which(apply(test.data, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
new_data = preProcessData(data[-outliers.train,],test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='knn')
scaled.train.outliers.knn$C <- scaled.train.knn$C
predictAndSave(scaled.train.outliers.knn, scaled.test, 'missing-knn-sin-outliers-raros-outliers-knn.csv', kernel="radial")



# 16
outliers.train <- which(apply(data[,-ncol(data)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
outliers.test <- which(apply(test.data, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
new_data = preProcessData(data[-outliers.train,],test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn[,1:ncol(scaled.train.knn)-1], type='knn')
scaled.train.outliers.knn$C <- scaled.train.knn$C
scaled.train.knn.balanced <- solveUnbalance(scaled.train.outliers.knn)
predictAndSave(scaled.train.knn.balanced, scaled.test,'missing-knn-sin-outliers-raros--balance.csv',kernel="radial")



# 17
# Elimino variables menos relacionadas con la salida
scaled.train.remove.low.cor <- removeLowCorrelationAttributes(scaled.train.knn)
predictAndSave(scaled.train.remove.low.cor,scaled.test,'20.csv')


# 18
pca <- prcomp(scaled.train.knn[,-ncol(scaled.train.knn)], center=T, scale=T)
train.pca <- as.data.frame(predict(pca, scaled.train.knn)[,1:30])
train.pca$C = scaled.train.knn$C
test.pca <- as.data.frame(predict(pca, scaled.test)[, 1:30])
predictAndSave(train.pca,test.pca,'22.csv')


# 19
if(length(outliers.train) > 0) train <- train[-outliers.train,]
train.completed <- computeMissingValues(train,type="knn") 
train.completed$C <- as.factor(train.completed$C)
train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
train.transformed <- attr.transform.add(train.cleaned)
scaler <- preProcess(train.transformed) # Centrado y escalado
train.scaled <- predict(scaler, train.transformed)
test[outliers.test,] <- 1
test.transformed <- attr.transform.add(test)
test.scaled <- predict(scaler, test.transformed)
predictAndSave(train.scaled,test.scaled,'23.csv')




# 20
train$C <- as.factor(data$C)
train.cleaned.2 <- CVCF(data, consensus = F)$cleanData
train.transformed.2 <- attr.transform.add(train.cleaned.2) %>% dplyr::select(-C,C)
train.completed.2 <- knnImputation(train.transformed.2)
scaler <- preProcess(train.completed.2)
train.scaled.2 <- predict(scaler, train.completed.2)

test[outliers.test,] <- 1 
test.transformed <- attr.transform.add(test)
test.scaled <- predict(scaler, test.transformed)

kernels <- c("linear","polynomial","radial","sigmoid")
i<-0
while ( i < length(kernels)){
  i<-i+1
  predictAndSave(train.scaled.2,test.scaled,paste(kernels[i],'24.csv'),kernels[i])  
}
