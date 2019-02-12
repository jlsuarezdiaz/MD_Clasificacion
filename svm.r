require('e1071')
set.seed(1234)

predictAndSave <- function(train, test,filename,kernel="radial"){
  model <- svm(C~.,data = train,kernel=kernel)
  pred <- predict(model,train)
  print(mean(round(pred)==train$C))
  #print(mean(pred==train$C))
  #pred <- predict(model,test)
  write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=round(pred)), file = filename, row.names=FALSE)
  #write.csv(data.frame('Id'=c(1:length(pred)),'Prediction'=pred), file = filename, row.names=FALSE)
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

scaled.train.rf = computeMissingValues(scaled.train, type = "rf")
write.csv(scaled.train, file = '/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/MD_Clasificacion/train-missing-rf.csv', row.names=FALSE)
predictAndSave(scaled.train.rf,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/rf-missing.csv')


# OUTLIERS
scaled.train.outliers.knn <- computeOutliers(scaled.train.knn, type='knn')
predictAndSave(scaled.train.outliers.knn,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/knn-scaled+outliers.csv')

scaled.train.outliers.mean <- computeOutliers(scaled.train.knn, type='mean')
predictAndSave(scaled.train.outliers.mean,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/knn-scaled-mean-outliers.csv')


scaled.train.outliers.rf <- computeOutliers(scaled.train.rf, type='rf')
# CLASS FROM {1,2} -> {0,1}
scaled.train.outliers.rf$C <- as.numeric(scaled.train.outliers.rf$C)
scaled.train.outliers.rf$C[scaled.train.outliers.rf$C==1] <- rep(0,length(scaled.train.outliers.rf$C[scaled.train.outliers.rf$C==1]))
scaled.train.outliers.rf$C[scaled.train.outliers.rf$C==2] <- rep(1,length(scaled.train.outliers.rf$C[scaled.train.outliers.rf$C==2]))
scaled.train.outliers.rf$C

predictAndSave(scaled.train.outliers.rf,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/rf-scaled-rf-outliers.csv')
write.csv(scaled.train.outliers.rf, file = '/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/MD_Clasificacion/train-missing-rf-outliers-rf.csv', row.names=FALSE)

scaled.train.no.noise <- filterNoiseData(scaled.train.knn)

# UNBALANCE
round(prop.table(table(data$C)) * 100, digits = 1)

i<- 1
scaled.train.outliers.rf.balanced <- solveUnbalance(scaled.train.outliers.rf)
predictAndSave(scaled.train.outliers.rf.balanced, scaled.test,paste('/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/',paste(balance.types[i],'-balance+rf.csv')))

# REMOVE COLUMNS WITH LOWER CORRELATION WITH OUTPUT
scaled.train.remove.low.cor <- removeLowCorrelationAttributes(scaled.train)
predictAndSave(scaled.train.remove.low.cor,scaled.test,'/Users/marta/Dropbox/MASTER/Minería de datos, preprocesamiento y clasificación/Trabajo/knn-scaled-mean-outliers-remove-low-cor.csv')

