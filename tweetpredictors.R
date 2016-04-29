#install.packages("readr")
#library(readr)
#install.packages("Matrix")
library(Matrix)
#install.packages("ggplot2")
library(ggplot2)
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
library(glmnet)
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
library(gbm)
#install.packages("e1071")
library(e1071)
#install.packages("tm")
library(tm)
#install.packages("gbm")
library(gbm)


##### Helpful Functions #####
# rm(list = ls())


##### Preliminaries #####
setwd("~/Dropbox/Spring 2016/stat154/tweetpredictors")

load("data/TrainTest.RData")
#load("data/.RData")
#load("data/gbm1.RData")
#load("data/gbm2.RData")
#load("data/rf.RData")

source("./data/ClassificationMetrics.R")

#data <- read_csv("data/MaskedDataRaw.csv")
words <- read.csv("data/vocab.csv", header = FALSE)
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

##### Exploratory Data Analysis #####
# how many times do each of the predictor show up in positive tweets and negative tweets
X.positive <- X.clean.bin[which(y.clean == 1),]
X.negative <- X.clean.bin[which(y.clean == 0),]

# uncleaned version for comparison
X.positive <- (X>0)[which(y == 1),]
X.negative <- (X>0)[which(y == 0),]

X.positive.popular <- apply(X.positive,2,sum)[order(apply(X.positive,2,sum), decreasing = TRUE)]
X.negative.popular <- apply(X.negative,2,sum)[order(apply(X.negative,2,sum), decreasing = TRUE)]

X.positive.popular[1:50]
X.negative.popular[1:50]


##### Model Fitting #####
colnames(X.full.cleaned) <- NULL #avoiding invalid multibyte string error
colnames(X.clean) <- NULL #avoiding invalid multibyte string error
colnames(X.clean.bin) <- NULL #avoiding invalid multibyte string error
set.seed(222222222)

set <- sample(nrow(X.clean), 10000)

# full predictors
X.full.train <- X.full.cleaned[-set,]
y.full.train <- y.full.cleaned[-set]
X.full.test <- X.full.cleaned[set,]
y.full.test <- y.full.cleaned[set]
df.train.full.cleaned <- data.frame(y.full.train, as.matrix(X.full.train))

# cleaned predictors
X.train <- X.clean[-set,]
y.train <- y.clean[-set]
X.test <- X.clean[set,]
y.test <- y.clean[set]
df.train.cleaned <- data.frame(y.train, as.matrix(X.train))

# binary predictors
X.train.bin <- X.clean.bin[-set,]
X.test.bin <- X.clean.bin[set,]
df.train.bin.cleaned <- data.frame(y.train, as.matrix(X.train.bin))

# penalized logreg full
subset.tweet.full.glm.sparse <- cv.glmnet(X.full.train, y.full.train, family = "binomial")
out.glm.full.sparse <- predict(subset.tweet.full.glm.sparse, X.full.test, type = "response") > .5
subset.tweet.full.glm.sparse$lambda.min
accuracy_score(y.full.test,out.glm.full.sparse)
f1_score(y.full.test,out.glm.full.sparse)

# penalized logreg cleaned
subset.tweet.glm.sparse <- cv.glmnet(X.train, y.train, family = "binomial")
out.glm.sparse <- predict(subset.tweet.glm.sparse, X.test, type = "response") > .5
subset.tweet.glm.sparse$lambda.min
accuracy_score(y.test,out.glm.sparse)
f1_score(y.test,out.glm.sparse)

# penalized logreg binary
subset.tweet.bin.glm.sparse <- cv.glmnet(X.train, y.train, family = "binomial")
out.glm.bin.sparse <- predict(subset.tweet.bin.glm.sparse, X.test.bin, type = "response") > .5
accuracy_score(y.test,out.glm.bin.sparse)
f1_score(y.test,out.glm.bin.sparse)


##### Plots for Writeup #####
# penalized logreg
ggplot(data.frame(subset.tweet.full.glm.sparse$lambda, subset.tweet.full.glm.sparse$cvm), aes(x=subset.tweet.full.glm.sparse$lambda, y=subset.tweet.full.glm.sparse$cvm))+geom_line()+ggtitle("Mean CV Error for Sparse Logistic Regression on Full Data")+xlab("Lambda")+ylab("CVM")
ggplot(data.frame(subset.tweet.glm.sparse$lambda, subset.tweet.glm.sparse$cvm), aes(x=subset.tweet.glm.sparse$lambda, y=subset.tweet.glm.sparse$cvm))+geom_line()+ggtitle("Mean CV Error for Sparse Logistic Regression on Cleaned Data")+xlab("Lambda")+ylab("CVM")
ggplot(data.frame(subset.tweet.bin.glm.sparse$lambda, subset.tweet.bin.glm.sparse$cvm), aes(x=subset.tweet.bin.glm.sparse$lambda, y=subset.tweet.bin.glm.sparse$cvm))+geom_line()+ggtitle("Mean CV Error for Sparse Logistic Regression on Binarized Data")+xlab("Lambda")+ylab("CVM")

# gbm
gbm1 <- data.frame(c(500, 1000, 1500, 2000), c(1-0.676, 1-0.7014, 1-0.7199, 1-0.7252))
names(gbm1) <- c("n.trees", "misclassif")
gbm2 <- data.frame(c(500, 1000, 1500, 2000), c(0.7303595, 0.7404381, 0.7477713, 0.7487658))
names(gbm2) <- c("n.trees", "f1")
ggplot(gbm1, aes(x = n.trees, y = misclassif)) + geom_line() + ggtitle("Misclassification Error Against the Number of Trees") + xlab("Trees")+ylab("Misclassification Error")
ggplot(gbm2, aes(x = n.trees, y = f1)) + geom_line() + ggtitle("F1 Score Against the Number of Trees") + xlab("Trees")+ylab("F1 Score")

# rf
rf1 <- data.frame(c(100, 500, 1000, 1500), c(1-0.7237, 1-0.7227, 1-0.7223, 1-0.7243))
names(rf1) <- c("n.trees", "misclassif")
rf2 <- data.frame(c(100, 500, 1000, 1500), c(0.7181475, 0.7168386, 0.7165459, 0.7182422))
names(rf2) <- c("n.trees", "f1")
ggplot(rf1, aes(x = n.trees, y = misclassif)) + geom_line() + ggtitle("Misclassification Error Against the Number of Trees") + xlab("Trees")+ylab("Misclassification Error")
ggplot(rf2, aes(x = n.trees, y = f1)) + geom_line() + ggtitle("F1 Score Against the Number of Trees") + xlab("Trees")+ylab("F1 Score")


##### Predicting On Xtest For Submission #####
kaggle1 <- (predict(subset.tweet.full.glm.sparse, data.frame(as.matrix(Xtest.clean)), type = "response") > .5)*1
kaggle <- cbind(1:50000, kaggle1)
### random forest
kaggle1 <- predict(subset.tweet.bin.rf.1500, data.frame(as.matrix(Xtest.clean)))
kaggle <- cbind(1:50000, as.numeric(kaggle1)-1)
colnames(kaggle) <- c("id", "y")

write.table(kaggle, "submission.csv", col.names = c("id", "y"), row.names = FALSE, sep = ",")

View(kaggle)






######## testing stuff out ########

foo <- X.full.test
colnames(foo) <- words$V1
foo <- foo[,!is.element(colnames(X),stopwords(kind="en"))]
foo <- foo[,!is.element(colnames(foo),union(letters, LETTERS))]
foo <- foo[,-grep('[[:punct:]]',colnames(foo))]
foo <- foo[,-grep('[[:digit:]]',colnames(foo))]
foo <- foo[,-c(14,18,84,91,150,197,213,252,268,278,334,352,403,416,474,502,587,590,624,627,677,683,729,759,790)]



x1 <- predict(subset.tweet.full.glm.sparse, Xtest, type = "response") > .5
x2 <- predict(subset.tweet.bin.rf.1500, data.frame(as.matrix(Xtest.clean)))
x3 <- predict(subset.tweet.bin.gbm.2000, data.frame(as.matrix(Xtest.clean)),n.trees = 2000, type = "response") > .5
x4 <- ((x1+(as.numeric(x2)-1)+x3)/3 > .5)*1
accuracy_score(y.full.test,x4)


set <- sample(nrow(X.full.cleaned), 10000)

# full predictors
X.full.train <- X.full.cleaned[-set,]
y.full.train <- y.full.cleaned[-set]
X.full.test <- X.full.cleaned[set,]
y.full.test <- y.full.cleaned[set]
#df.train.full.cleaned <- data.frame(y.full.train, as.matrix(X.full.train))

# penalized logreg full
x1 <- cv.glmnet(X.full.train, y.full.train, family = "binomial")
x2 <- cv.glmnet(X.full.train, y.full.train, family = "binomial")
x3 <- cv.glmnet(X.full.train, y.full.train, family = "binomial")
x4 <- cv.glmnet(X.full.train, y.full.train, family = "binomial")
x5 <- cv.glmnet(X.full.train, y.full.train, family = "binomial")
y1 <- predict(x1, Xtest, type = "response") > .5
y2 <- predict(subset.tweet.bin.rf.1500, data.frame(as.matrix(Xtest)))
y3 <- predict(subset.tweet.bin.gbm.2000, data.frame(as.matrix(X.test.bin)),n.trees = 2000, type = "response") > .5
y4 <- predict(x4, Xtest, type = "response") > .5
y5 <- predict(x5, Xtest, type = "response") > .5



y6 <- (y1+y2+y3+y4+y5)>.5

kaggle <- cbind(1:50000, x4)
colnames(kaggle) <- c("id", "y")

write.table(kaggle, "submission.csv", col.names = c("id", "y"), row.names = FALSE, sep = ",")

View(kaggle)









