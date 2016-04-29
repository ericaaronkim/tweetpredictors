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
#library(randomForest)
#install.packages("rpart.plot")
#library(rpart.plot)
#install.packages("gbm")
#library(gbm)
#install.packages("e1071")
#library(e1071)
#install.packages("tm")
library(tm)
#install.packages("gbm")
library(gbm)


##### Helpful Function #####
# rm(list = ls())


##### Preliminaries #####
#setwd("~/Dropbox/Spring 2016/stat154/tweetpredictors")

#load("TrainTest.RData")

source("./ClassificationMetrics.R")

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


### 500 Trees Interaction Depth 5 ###
# gbm full
#subset.tweet.full.gbm <- gbm(y.full.train ~ ., data = df.train.full.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=500, shrinkage=.01, verbose=TRUE)
#subset.tweet.full.gbm.predict <- predict(subset.tweet.full.gbm, data.frame(as.matrix(X.full.test)), type = "response", n.trees = 500) > .5
#accuracy_score(y.full.test,subset.tweet.full.gbm.predict)
#f1_score(y.full.test,subset.tweet.full.gbm.predict)

# gbm cleaned
#subset.tweet.gbm <- gbm(y.full.train ~ ., data = df.train.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=500, shrinkage=.01, verbose=TRUE)
#subset.tweet.gbm.predict <- predict(subset.tweet.gbm, data.frame(as.matrix(X.test)), type = "response", n.trees = 500) > .5
#accuracy_score(y.test,subset.tweet.gbm.predict)
#f1_score(y.test,subset.tweet.gbm.predict)

# gbm binary
#subset.tweet.bin.gbm.500 <- gbm(y.train ~ ., data = df.train.bin.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=500, shrinkage=.01, verbose=TRUE)
#subset.tweet.bin.gbm.predict.500 <- predict(subset.tweet.bin.gbm.500, data.frame(as.matrix(X.test.bin)),n.trees = 500, type = "response") > .5
#accuracy_score(y.test,subset.tweet.bin.gbm.predict.500) # 0.676
#f1_score(y.test,subset.tweet.bin.gbm.predict.500) # 0.7303595

### 1000 Trees Interaction Depth 5 ###
# gbm full
#subset.tweet.full.gbm <- gbm(y.full.train ~ ., data = df.train.full.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=1000, shrinkage=.01, verbose=TRUE)
#subset.tweet.full.gbm.predict <- predict(subset.tweet.full.gbm, data.frame(as.matrix(X.full.test)), type = "response", n.trees = 1000) > .5
#accuracy_score(y.full.test,subset.tweet.full.gbm.predict)
#f1_score(y.full.test,subset.tweet.full.gbm.predict)

# gbm cleaned
#subset.tweet.gbm <- gbm(y.full.train ~ ., data = df.train.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=1000, shrinkage=.01, verbose=TRUE)
#subset.tweet.gbm.predict <- predict(subset.tweet.gbm, data.frame(as.matrix(X.test)), type = "response", n.trees = 1000) > .5
#accuracy_score(y.test,subset.tweet.gbm.predict)
#f1_score(y.test,subset.tweet.gbm.predict)

# gbm binary
#subset.tweet.bin.gbm.1000 <- gbm(y.train ~ ., data = df.train.bin.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=1000, shrinkage=.01, verbose=TRUE)
#subset.tweet.bin.gbm.predict.1000 <- predict(subset.tweet.bin.gbm.1000, data.frame(as.matrix(X.test.bin)),n.trees = 1000, type = "response") > .5
#accuracy_score(y.test,subset.tweet.bin.gbm.predict.1000) # 0.7014
#f1_score(y.test,subset.tweet.bin.gbm.predict.1000) # 0.7404381


### 1500 Trees Interaction Depth 5 ###
# gbm full
#subset.tweet.full.gbm <- gbm(y.full.train ~ ., data = df.train.full.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=1000, shrinkage=.01, verbose=TRUE)
#subset.tweet.full.gbm.predict <- predict(subset.tweet.full.gbm, data.frame(as.matrix(X.full.test)), type = "response", n.trees = 1000) > .5
#accuracy_score(y.full.test,subset.tweet.full.gbm.predict)
#f1_score(y.full.test,subset.tweet.full.gbm.predict)

# gbm cleaned
#subset.tweet.gbm <- gbm(y.full.train ~ ., data = df.train.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=1000, shrinkage=.01, verbose=TRUE)
#subset.tweet.gbm.predict <- predict(subset.tweet.gbm, data.frame(as.matrix(X.test)), type = "response", n.trees = 1000) > .5
#accuracy_score(y.test,subset.tweet.gbm.predict)
#f1_score(y.test,subset.tweet.gbm.predict)

# gbm binary
#subset.tweet.bin.gbm.1500 <- gbm(y.train ~ ., data = df.train.bin.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=1500, shrinkage=.01, verbose=TRUE)
#subset.tweet.bin.gbm.predict.1500 <- predict(subset.tweet.bin.gbm.1500, data.frame(as.matrix(X.test.bin)),n.trees = 1500, type = "response") > .5
#accuracy_score(y.test,subset.tweet.bin.gbm.predict.1500) # .7199 (see gbm2.RData)
#f1_score(y.test,subset.tweet.bin.gbm.predict.1500) # 0.7477713 (see gbm2.RData)


### 2000 Trees Interaction Depth 5 ###
# gbm full
subset.tweet.full.gbm <- gbm(y.full.train ~ ., data = df.train.full.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=2000, shrinkage=.01, verbose=TRUE)
subset.tweet.full.gbm.predict <- predict(subset.tweet.full.gbm, data.frame(as.matrix(X.full.test)), type = "response", n.trees = 2000) > .5
accuracy_score(y.full.test,subset.tweet.full.gbm.predict) # (see gbm2.RData)
f1_score(y.full.test,subset.tweet.full.gbm.predict) # (see gbm2.RData)

# gbm cleaned
<<<<<<< HEAD
subset.tweet.gbm <- gbm(y.train ~ ., data = df.train.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=2000, shrinkage=.01, verbose=TRUE)
=======
subset.tweet.gbm <- gbm(y.full.train ~ ., data = df.train.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=2000, shrinkage=.01, verbose=TRUE)
>>>>>>> 065668bb1a3aca3aed88c31267859e6b305b757e
subset.tweet.gbm.predict <- predict(subset.tweet.gbm, data.frame(as.matrix(X.test)), type = "response", n.trees = 2000) > .5
accuracy_score(y.test,subset.tweet.gbm.predict) # (see gbm2.RData)
f1_score(y.test,subset.tweet.gbm.predict) # (see gbm2.RData)

# gbm binary
#subset.tweet.bin.gbm.2000 <- gbm(y.train ~ ., data = df.train.bin.cleaned, distribution = "adaboost", interaction.depth=5, n.trees=2000, shrinkage=.01, verbose=TRUE)
#subset.tweet.bin.gbm.predict.2000 <- predict(subset.tweet.bin.gbm.2000, data.frame(as.matrix(X.test.bin)),n.trees = 2000, type = "response") > .5
#accuracy_score(y.test,subset.tweet.bin.gbm.predict.2000) # 0.7252 (see gbm2.RData)
#f1_score(y.test,subset.tweet.bin.gbm.predict.2000) # 0.7487658 (see gbm2.RData)




























