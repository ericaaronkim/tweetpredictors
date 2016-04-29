#install.packages("readr")
#library(readr)
#install.packages("Matrix")
library(Matrix)
#install.packages("ggplot2")
#library(ggplot2)
#install.packages("kernlab")
#library(kernlab)
#install.packages("mclust")
#library(mclust)
#install.packages("Rtsne")
#library(Rtsne)
#install.packages("cluster")
#library(cluster)
#install.packages("fpc")
#library(fpc)
#install.packages("fGarch")
#library(fGarch)
#install.packages("lattice")
#library(lattice)
#install.packages("quantreg")
#library(quantreg)
#install.packages("FNN")
#library(FNN)
#install.packages("MASS")
#library(MASS)
#install.packages("glmnet")
#library(glmnet)
#install.packages("lars")
#library(lars)
#install.packages("fda.usc")
#library(fda.usc)
#install.packages("pls")
#library(pls)
#install.packages("spls")
#library(spls)
#install.packages("penalized")
#library(penalized)
#install.packages("boot")
#library(boot)
#install.packages("ipred")
#library(ipred)
#install.packages("rpart")
#library(rpart)
#install.packages("randomForest")
library(randomForest)
#install.packages("rpart.plot")
#library(rpart.plot)
#install.packages("gbm")
#library(gbm)
#install.packages("e1071")
#library(e1071)
#install.packages("tm")
library(tm)


##### Helpful Function #####
# rm(list = ls())


##### Preliminaries #####
#setwd("~/Dropbox/Spring 2016/stat154/tweetpredictors")

load("TrainTest.RData")

source(".ClassificationMetrics.R")

#data <- read_csv("MaskedDataRaw.csv")
words <- read.csv("vocab.csv", header = FALSE)
colnames(X) <- words$V1


##### Data Cleaning #####
# identifying white noise
#which(apply(X, 1, sum) == 0)
#length(which(apply(X, 1, sum) == 0))

# cleaning stopwords, single letters, punctuation, numbers, and unicode
X.clean <- X
Xtest.clean <- Xtest[,!is.element(colnames(X),stopwords(kind="en"))]
X.clean <- X[,!is.element(colnames(X),stopwords(kind="en"))]
Xtest.clean <- Xtest.clean[,!is.element(colnames(X.clean),union(letters, LETTERS))]
X.clean <- X.clean[,!is.element(colnames(X.clean),union(letters, LETTERS))]
Xtest.clean <- Xtest.clean[,-grep('[[:punct:]]',colnames(X.clean))]
X.clean <- X.clean[,-grep('[[:punct:]]',colnames(X.clean))]
Xtest.clean <- Xtest.clean[,-grep('[[:digit:]]',colnames(X.clean))]
X.clean <- X.clean[,-grep('[[:digit:]]',colnames(X.clean))]
Xtest.clean <- Xtest.clean[,-c(14,18,84,91,150,197,213,252,268,278,334,352,403,416,474,502,587,590,624,627,677,683,729,759,790)] #manually found unicode
X.clean <- X.clean[,-c(14,18,84,91,150,197,213,252,268,278,334,352,403,416,474,502,587,590,624,627,677,683,729,759,790)] #manually found unicode

# full X removed rows
X.full.cleaned <- X[which(apply(X, 1, sum) != 0),]
y.full.cleaned <- y[which(apply(X, 1, sum) != 0)]
Xtest.full.cleaned <- Xtest[which(apply(X, 1, sum) != 0)]

# remove rows with all 0
zero.rows <- which(apply(X.clean, 1, sum) != 0)
X.clean <- X.clean[zero.rows,]
y.clean <- y[zero.rows]

# binary data (the existence of a word instead of the quantity of the word)
X.clean.bin <- (X.clean>0)*1


##### Model Fitting #####
colnames(X.full.cleaned) <- NULL #avoiding invalid multibyte string error
colnames(X.clean) <- NULL #avoiding invalid multibyte string error
colnames(X.clean.bin) <- NULL #avoiding invalid multibyte string error
set.seed(222222222)

set <- sample(nrow(X.clean), nrow(X.clean)-10000)

# full predictors
X.full.train <- X.full.cleaned[set,]
y.full.train <- y.full.cleaned[set]
X.full.test <- X.full.cleaned[-set,]
y.full.test <- y.full.cleaned[-set]
df.train.full.cleaned <- data.frame(y.full.train, as.matrix(X.full.train))

# cleaned predictors
X.train <- X.clean[set,]
y.train <- y.clean[set]
X.test <- X.clean[-set,]
y.test <- y.clean[-set]
df.train.cleaned <- data.frame(y.train, as.matrix(X.train))

# binary predictors
X.train.bin <- X.clean.bin[set,]
X.test.bin <- X.clean.bin[-set,]
df.train.bin.cleaned <- data.frame(y.train, as.matrix(X.train.bin))

### 100 Trees ###
# random forest full
subset.tweet.full.rf.100 <- randomForest(df.train.full.cleaned[,2:ncol(df.train.full.cleaned)], as.factor(df.train.full.cleaned[,1]), ntrees = 100)
subset.tweet.full.rf.prediction.100 <- predict(subset.tweet.full.rf.100, data.frame(as.matrix(X.full.test)))
accuracy_score(y.full.test,subset.tweet.full.rf.prediction.100)
f1_score(y.full.test,subset.tweet.full.rf.prediction.100)

# random forest cleaned
subset.tweet.rf.100 <- randomForest(df.train.cleaned[,2:ncol(df.train.cleaned)], as.factor(df.train.cleaned[,1]), ntrees = 100)
subset.tweet.rf.prediction.100 <- predict(subset.tweet.rf.100, data.frame(as.matrix(X.test)))
accuracy_score(y.test,subset.tweet.rf.prediction.100)
f1_score(y.test,subset.tweet.rf.prediction.100)

# random forest binary
#subset.tweet.bin.rf.100 <- randomForest(df.train.bin.cleaned[,2:ncol(df.train.bin.cleaned)], as.factor(df.train.bin.cleaned[,1]), ntrees = 100)
#subset.tweet.bin.rf.prediction.100 <- predict(subset.tweet.bin.rf.100, data.frame(as.matrix(X.test.bin)))
#accuracy_score(y.test,subset.tweet.bin.rf.prediction.100) # 0.7237
#f1_score(y.test,subset.tweet.bin.rf.prediction.100) # 0.7181475

### 500 Trees ###
# random forest full
#subset.tweet.full.rf.500 <- randomForest(df.train.full.cleaned[,2:ncol(df.train.full.cleaned)], as.factor(df.train.full.cleaned[,1]), ntrees = 500)
#subset.tweet.full.rf.prediction.500 <- predict(subset.tweet.full.rf.500, data.frame(as.matrix(X.full.test)))
#accuracy_score(y.full.test,subset.tweet.full.rf.prediction.500)
#f1_score(y.full.test,subset.tweet.full.rf.prediction.500)

# random forest cleaned
#subset.tweet.rf.500 <- randomForest(df.train.cleaned[,2:ncol(df.train.cleaned)], as.factor(df.train.cleaned[,1]), ntrees = 500)
#subset.tweet.rf.prediction.500 <- predict(subset.tweet.rf.500, data.frame(as.matrix(X.test)))
#accuracy_score(y.test,subset.tweet.rf.prediction.500)
#f1_score(y.test,subset.tweet.rf.prediction.500)

# random forest binary
#subset.tweet.bin.rf.500 <- randomForest(df.train.bin.cleaned[,2:ncol(df.train.bin.cleaned)], as.factor(df.train.bin.cleaned[,1]), ntrees = 500)
#subset.tweet.bin.rf.prediction.500 <- predict(subset.tweet.bin.rf.500, data.frame(as.matrix(X.test.bin)))
#accuracy_score(y.test,subset.tweet.bin.rf.prediction.500) # 0.7227
#f1_score(y.test,subset.tweet.bin.rf.prediction.500) # 0.7168386

### 1000 Trees ###
# random forest full
#subset.tweet.full.rf.1000 <- randomForest(df.train.full.cleaned[,2:ncol(df.train.full.cleaned)], as.factor(df.train.full.cleaned[,1]), ntrees = 1000)
#subset.tweet.full.rf.prediction.1000 <- predict(subset.tweet.full.rf.1000, data.frame(as.matrix(X.full.test)))
#accuracy_score(y.full.test,subset.tweet.full.rf.prediction.1000)
#f1_score(y.full.test,subset.tweet.full.rf.prediction.1000)

# random forest cleaned
#subset.tweet.rf.1000 <- randomForest(df.train.cleaned[,2:ncol(df.train.cleaned)], as.factor(df.train.cleaned[,1]), ntrees = 1000)
#subset.tweet.rf.prediction.1000 <- predict(subset.tweet.rf.1000, data.frame(as.matrix(X.test)))
#accuracy_score(y.test,subset.tweet.rf.prediction.1000)
#f1_score(y.test,subset.tweet.rf.prediction.1000)

# random forest binary
#subset.tweet.bin.rf.1000 <- randomForest(df.train.bin.cleaned[,2:ncol(df.train.bin.cleaned)], as.factor(df.train.bin.cleaned[,1]), ntrees = 1000)
#subset.tweet.bin.rf.prediction.1000 <- predict(subset.tweet.bin.rf.1000, data.frame(as.matrix(X.test.bin)))
#accuracy_score(y.test,subset.tweet.bin.rf.prediction.1000) # 0.7223
#f1_score(y.test,subset.tweet.bin.rf.prediction.1000) # 0.7165459

### 1500 Trees ###
# random forest full
#subset.tweet.full.rf.1500 <- randomForest(df.train.full.cleaned[,2:ncol(df.train.full.cleaned)], as.factor(df.train.full.cleaned[,1]), ntrees = 1500)
#subset.tweet.full.rf.prediction.1500 <- predict(subset.tweet.full.rf.1500, data.frame(as.matrix(X.full.test)))
#accuracy_score(y.full.test,subset.tweet.full.rf.prediction.1500)
#f1_score(y.full.test,subset.tweet.full.rf.prediction.1500)

# random forest cleaned
#subset.tweet.rf.1500 <- randomForest(df.train.cleaned[,2:ncol(df.train.cleaned)], as.factor(df.train.cleaned[,1]), ntrees = 1500)
#subset.tweet.rf.prediction.1500 <- predict(subset.tweet.rf.1500, data.frame(as.matrix(X.test)))
#accuracy_score(y.test,subset.tweet.rf.prediction.1500)
#f1_score(y.test,subset.tweet.rf.prediction.1500)

# random forest binary
#subset.tweet.bin.rf.1500 <- randomForest(df.train.bin.cleaned[,2:ncol(df.train.bin.cleaned)], as.factor(df.train.bin.cleaned[,1]), ntrees = 1500)
#subset.tweet.bin.rf.prediction.1500 <- predict(subset.tweet.bin.rf.1500, data.frame(as.matrix(X.test.bin)))
#accuracy_score(y.test,subset.tweet.bin.rf.prediction.1500) # 0.7243
#f1_score(y.test,subset.tweet.bin.rf.prediction.1500) # 0.7182422





















