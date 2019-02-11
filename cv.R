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
