require('e1071')
set.seed(1234)

predictAndSave <- function(train, test,filename,kernel="radial"){
  model <- svm(C~.,data = train,kernel=kernel)
  pred <- predict(model,train)
  print(pred)
  #pred <- round(pred)
  print(mean(pred==train$C))
  pred <- predict(model,test)
  #write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=round(pred)), file = filename, row.names=FALSE)
  write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=pred), file = filename, row.names=FALSE)
  #return(pred)
}


data <- read.csv("/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/train.csv", header=TRUE, na.strings="?")
test.data <- read.csv("/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/test.csv", header=TRUE, na.strings="?")

# CENTER AND SCALE DATA
new_data = preProcessData(data,test.data)
scaled.train = new_data[[1]]
scaled.test = new_data[[2]]

# FILL MISSING VALUES
scaled.train.knn = computeMissingValues(scaled.train, type = "knn")
predictAndSave(scaled.train,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/knn-scaled.csv')

scaled.train = computeMissingValues(scaled.train, type = "rf")
write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=round(pred)), file = filename, row.names=FALSE)
predictAndSave(scaled.train,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/rf-missing.csv')


# OUTLIERS
scaled.train.outliers.knn <- computeOutliers(scaled.train, type='knn')
predictAndSave(scaled.train.outliers.knn,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/knn-scaled+outliers.csv')

scaled.train.outliers.mean <- computeOutliers(scaled.train.knn, type='mean')
predictAndSave(scaled.train.outliers.mean,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/knn-scaled-mean-outliers.csv')


# UNBALANCE
round(prop.table(table(data$C)) * 100, digits = 1)
balance.types = c('ubOver', 'ubUnder', 'ubSMOTE')
i <- 1
while( i < length(balance.types)){
  scaled.train.outliers.mean.balanced <- solveUnbalance(scaled.train.outliers.mean,type=balance.types[i])
  predictAndSave(scaled.train.outliers.mean.balanced, scaled.test,paste(balance.types[i],'-balance.csv'))
}


# REMOVE COLUMNS WITH LOWER CORRELATION WITH OUTPUT
scaled.train.remove.low.cor <- removeLowCorrelationAttributes(scaled.train)
predictAndSave(scaled.train.remove.low.cor,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/knn-scaled-mean-outliers-remove-low-cor.csv')


# KERNELS
kernels <- c("radial","polynomial")
i<-1
while( i< length(kernels)){
  predictAndSave(scaled.train.outliers.mean, scaled.test,paste(kernels[i],'.csv'),kernel=kernels[i])
  i<- i+1
}
