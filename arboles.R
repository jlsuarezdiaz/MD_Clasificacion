# lectura de datos
library("tree")
library(rpart)
train <- read.csv("./train.csv", header=TRUE, na.strings="?")
test <- read.csv("./test.csv", header = TRUE, na.strings = "?")

# 1 

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


# 2

# preprocesamiento

train = computeMissingValues(train, type = "knn", k = 1)
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


# 3

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


# 4 


# preprocesamiento
ini = mice(train, maxit = 0)
quitar = as.character(ini$loggedEvents[,"out"])
algo = mice(train, meth="pmm", seed = 500, remove_collinear = FALSE)
compData = complete(algo,1)
length(which(is.na(compData) == TRUE))
train = compData

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
write.csv(data.frame('Id'=c(1:length(model.pred)),'Prediction'=model.pred), file = paste("MedianCS.csv"), row.names=FALSE)


# 5

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


# 6 

# preprocesamiento 

train = computeMissingValues(train, type = "median")

train = computeOutliers(train, type = 'median')

boxplot(train)
summary(train[,16])

new_data = preProcessData(train,test)
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
