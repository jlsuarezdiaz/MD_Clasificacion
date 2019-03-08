source("cv.R")
source("preprocesado.R")
require('e1071')

predictAndSave <- function(train, test,filename,kernel="radial"){
  model <- svm(C~.,data = train,kernel=kernel)
  
  pred <- predict(model,train)
  print(mean(pred==train$C))
  #print(mean(round(pred)==train$C))
  
  pred <- predict(model,test)
  pred[outliers.test.por.la.cara] <- 0
  #write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=round(pred)), file = filename, row.names=FALSE)
  write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=pred), file = filename, row.names=FALSE)
  #return(pred)
}

data <- read.csv("train.csv", header=TRUE, na.strings="?")
test.data <- read.csv("test.csv", header=TRUE, na.strings="?")


# 1
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train = computeMissingValues(scaled.train, type = "remove")
predictAndSave(scaled.train,scaled.test,'subs-svm/1.csv')


# 2
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "mean")
predictAndSave(scaled.train,scaled.test,'subs-svm/2.csv')


# 3
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "mean")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='remove')
predictAndSave(scaled.train.outliers.knn,scaled.test,'subs-svm/3.csv')


# 4
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "mean")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='knn')
predictAndSave(scaled.train,scaled.test,'subs-svm/3.csv')


# 5
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
predictAndSave(scaled.train,scaled.test,'subs-svm/5.csv')



# 6
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='knn')
predictAndSave(scaled.train,scaled.test,'subs-svm/6.csv')



# 7
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='mean')
predictAndSave(scaled.train,scaled.test,'subs-svm/7.csv')



# 8
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='mean')
scaled.train.remove.low.cor <- removeLowCorrelationAttributes(scaled.train.outliers.knn)
predictAndSave(scaled.train.remove.low.cor,scaled.test,'subs-svm/8.csv')


# 9
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers <- computeOutliers(scaled.train, type='mean')
scaled.train.outliers.balanced <- solveUnbalance(scaled.train.outliers, "ubOver")
predictAndSave(scaled.train.outliers.balanced,scaled.test,'subs-svm/9.csv')


# 10
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='mean')
scaled.train.outliers.knn.balanced <- solveUnbalance(scaled.train.outliers.knn, "ubUndersampling")
predictAndSave(scaled.train.outliers.knn.balanced,scaled.test,'subs-svm/10.csv')



# 11
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers <- computeOutliers(scaled.train, type='mean')
scaled.train.outliers.balanced <- solveUnbalance(scaled.train.outliers, "ubSMOTE")
predictAndSave(scaled.train.outliers.balanced,scaled.test,'subs-svm/11.csv')



# 12
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='rf')
predictAndSave(scaled.train.outliers.knn,scaled.test,'subs-svm/12.csv')




# 13
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers <- computeOutliers(scaled.train, type='rf')
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
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='knn')
predictAndSave(scaled.train.outliers.knn, scaled.test, 'missing-knn-sin-outliers-raros-outliers-knn.csv', kernel="radial")



# 16
outliers.train <- which(apply(data[,-ncol(data)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
outliers.test <- which(apply(test.data, MARGIN=1, function(x) any(!is.na(x) && x < -68000)))
new_data = preProcessData(data[-outliers.train,],test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='knn')
scaled.train.knn.balanced <- solveUnbalance(scaled.train.knn)
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
outliers.train.por.la.cara <- which(apply(data[,-ncol(train)], MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
outliers.test.por.la.cara <- which(apply(test, MARGIN=1, function(x) any(!is.na(x) & x < -68000)))
if(length(outliers.train.por.la.cara) > 0) train <- train[-outliers.train.por.la.cara,]
train.completed <- computeMissingValues(train,type="knn") 
train.completed$C <- as.factor(train.completed$C)
train.cleaned <- CVCF(train.completed, consensus = F)$cleanData
train.transformed <- attr.transform.add(train.cleaned)
scaler <- preProcess(train.transformed) # Centrado y escalado
train.scaled <- predict(scaler, train.transformed)
test[outliers.test.por.la.cara,] <- 1
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

test[outliers.test.por.la.cara,] <- 1 
test.transformed <- attr.transform.add(test)
test.scaled <- predict(scaler, test.transformed)

kernels <- c("linear","polynomial","radial","sigmoid")
i<-0
while ( i < length(kernels)){
  i<-i+1
  predictAndSave(train.scaled.2,test.scaled,paste(kernels[i],'24.csv'),kernels[i])  
}
