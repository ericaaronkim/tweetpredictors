library(readr)
library(Matrix)
library(ggplot2)
library(glmnet)
library(penalized)
library(randomForest)
library(gbm)
library(e1071)
library(tm)
library(neuralnet)


##### Preliminaries #####
setwd('~/Documents/class/stat154/tweetpredictors/')

load("data/TrainTest.RData")

source("data/ClassificationMetrics.R")

data <- read.csv("data/MaskedDataRaw.csv")
words <- read.csv("data/vocab.csv", header = FALSE)
colnames(X) <- words$V1


##### Data Cleaning #####
# cleaning stopwords, single letters, punctuation, numbers, and unicode
Xtest <- Xtest[,!is.element(colnames(X),stopwords(kind="en"))]
X.clean <- X[,!is.element(colnames(X),stopwords(kind="en"))]
Xtest <- Xtest[,!is.element(colnames(X.clean),union(letters, LETTERS))]
X.clean <- X.clean[,!is.element(colnames(X.clean),union(letters, LETTERS))]
Xtest <- Xtest[,-grep('[[:punct:]]',colnames(X.clean))]
X.clean <- X.clean[,-grep('[[:punct:]]',colnames(X.clean))]
Xtest <- Xtest[,-grep('[[:digit:]]',colnames(X.clean))]
X.clean <- X.clean[,-grep('[[:digit:]]',colnames(X.clean))]
Xtest <- Xtest[,-c(14,18,84,91,150,197,213,252,268,278,334,352,403,416,474,502,587,590,624,627,677,683,729,759,790)] #manually found unicode
X.clean <- X.clean[,-c(14,18,84,91,150,197,213,252,268,278,334,352,403,416,474,502,587,590,624,627,677,683,729,759,790)] #manually found unicode

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
colnames(X.clean) <- NULL #avoiding invalid multibyte string error
colnames(X.clean.bin) <- NULL #avoiding invalid multibyte string error
set.seed(222222222)

set <- sample(nrow(X.clean), 40000)

# regular predictors
X.train <- X.clean[set,]
y.train <- y.clean[set]
X.test <- X.clean[-set,]
y.test <- y.clean[-set]
df.train.cleaned <- data.frame(y.train, as.matrix(X.train))

# binary predictors
X.train.bin <- X.clean.bin[set,]
X.test.bin <- X.clean.bin[-set,]
df.train.bin.cleaned <- data.frame(y.train, as.matrix(X.train.bin))

# random forest regular
subset.tweet.rf <- randomForest(df.train.cleaned[,2:ncol(df.train.cleaned)], as.factor(df.train.cleaned[,1]), ntrees = 1000)
subset.tweet.rf.prediction <- predict(subset.tweet.rf, data.frame(as.matrix(X.test)))
accuracy_score(y.test,subset.tweet.rf.prediction)

# random forest binary
subset.tweet.bin.rf <- randomForest(df.train.bin.cleaned[,2:ncol(df.train.bin.cleaned)], as.factor(df.train.bin.cleaned[,1]), ntrees = 1000)
subset.tweet.bin.rf.prediction <- predict(subset.tweet.rf, data.frame(as.matrix(X.test.bin)))
accuracy_score(y.test,subset.tweet.bin.rf.prediction)

# logreg regular
subset.tweet.glm <- glm(y.train ~ ., data = df.train.cleaned, family = "binomial")
out.glm <- predict.glm(subset.tweet.glm, data.frame(as.matrix(X.test)), type = "response")
out.glm <- (out.glm >= 0.5)
accuracy_score(y.test,out.glm)

# logreg binary
subset.tweet.bin.glm <- glm(y.train ~ ., data = df.train.bin.cleaned, family = "binomial")
out.glm.bin <- predict.glm(subset.tweet.bin.glm, data.frame(as.matrix(X.test.bin)), type = "response")
out.glm.bin <- (out.glm.bin >= 0.5)
accuracy_score(y.test,out.glm)

# penalized logreg regular
subset.tweet.glm.sparse <- cv.glmnet(X.train, y.train, family = "binomial")
out.glm.sparse <- predict(subset.tweet.glm.sparse, X.test, type = "response") > .5
accuracy_score(y.test,out.glm.sparse)

# penalized logreg binary
subset.tweet.bin.glm.sparse <- cv.glmnet(X.train.bin, y.train, family = "binomial")
out.glm.bin.sparse <- predict(subset.tweet.glm.sparse, X.test.bin, type = "response") > .5
accuracy_score(y.test,out.glm.bin.sparse)

# neural net
subset.tweet.nn.sparse = neuralnet(as.formula(paste('y.train~', paste(names(df.train.cleaned)[2:785], 
                                   collapse='+'))), data=df.train.cleaned, hidden=c(500), 
                                   linear.output=FALSE)
out.nn.sparse = compute(neural_net, X.test)$net.result > 0.5
accuracy_score(y.test, out.nn.sparse)

subset.tweet.nn.sparse.200 = neuralnet(as.formula(paste('y.train~', paste(names(df.train.bin.cleaned)[2:785], 
                                       collapse='+'))), data=df.train.bin.cleaned, hidden=c(200), 
                                       linear.output=FALSE)

subset.tweet.nn.sparse.400 = neuralnet(as.formula(paste('y.train~', paste(names(df.train.cleaned)[2:785], 
                                       collapse='+'))), data=df.train.cleaned, hidden=c(400), 
                                       linear.output=FALSE)

subset.tweet.nn.sparse.600 = neuralnet(as.formula(paste('y.train~', paste(names(df.train.cleaned)[2:785], 
                                       collapse='+'))), data=df.train.cleaned, hidden=c(600), 
                                       linear.output=FALSE)

##### Predicting On Xtest For Submission #####
kaggle1 <- predict(subset.tweet.bin.glm.sparse, Xtest, type = "response")
kaggle <- cbind(1:50000, kaggle1)
### random forest
kaggle1 <- predict(subset.tweet.rf, data.frame(as.matrix(Xtest)))
kaggle <- cbind(1:50000, as.numeric(kaggle1)-1)
colnames(kaggle) <- c("id", "y")

write.table(kaggle, "submission.csv", col.names = c("id", "y"), row.names = FALSE, sep = ",")

View(kaggle)
