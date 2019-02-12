train <- read.csv("train.csv", na.strings = c("?", "NA", "NR", "na", "NaN", "nan"))
test <- read.csv("test.csv", na.strings = c("?", "NA", "NR", "na", "NaN", "nan"))
sample <- read.csv("sampleSubmission.csv")

## BIBLIOTECAS

library(ggplot2)
library(caret)
library(RKEEL)
# library(rDML) # Por si acaso
library(kknn)
library(GGally)
library(Hmisc)
library(dplyr)

## ANÁLISIS

# Resumen de los datos
summary(train)
summary(test)

describe(train)
describe(test)

# Primera cosa rara: hay datos basura con -68000 y pico en todas sus variables.



## SUBIDA 1

set.seed(28)




## SUBIDA 2

set.seed(28)