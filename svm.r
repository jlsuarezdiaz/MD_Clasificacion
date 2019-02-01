require('e1071')
set.seed(1234)

data <- read.csv("/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/train.csv", header=TRUE, na.strings="?")
test.data <- read.csv("/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/test.csv", header=TRUE, na.strings="?")

# CENTER AND SCALE DATA
require(caret)
n <- length(data)
valoresPreprocesados <- caret::preProcess(data[1:n-1],method=c("center" ,"scale") )
data.scaled <- predict(valoresPreprocesados,data[1:n-1])
data.scaled <- cbind(data.scaled,C=data$C)


predictAndSave <- function(train, test,filename,kernel="radial"){
  model <- svm(C~.,data = train,kernel=kernel)
  pred <- predict(model,train)
  pred <- round(pred)
  print(mean(pred==train$C))
  pred <- predict(model,test)
  write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=round(pred)), file = filename, row.names=FALSE)
  pred
}

# FILL MISSING VALUES
train.data <- computeMissingValues(data.scaled,type='knn',1)

# OUTLIERS
k <- 1
a <- computeOutliers(train.data, type='knn', k = k)
predictAndSave(a,test.data,paste(paste('./outliers-knn-',k),'.csv'))



# KERNELS
kernels <- c("linear","polynomial","radial basis","sigmoid")
i<-1
while( i< length(kernels)){
  predictAndSave(train, test,paste(kernels[i],'.csv'),kernel=kernels[i])
  i<- i+1
}
predictAndSave(train, test,paste('BORRAME','.csv'))
