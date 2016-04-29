library(glmnet)
library(randomForest)
library(gbm)
library(neuralnet)
library(tm)

setwd('~/Documents/class/stat154/tweetpredictors/')
load('data/TrainTest.RData')
load('data/models.RData')
load('data/functions.RData')

source('data/ClassificationMetrics.R')

##### Data Cleaning #####
data.full = clean_rows(X, y)
X.full = binarize_data(data.full$X)
y.full = data.full$y

val_ind = sample(49712, 10000)
val_pred = as.matrix(X.full[val_ind,])
val_pred_sparse = clean_features(val_pred)
val_label = y.full[val_ind]
df.train <- data.frame(val_label, as.matrix(val_pred))


##### Model Fitting #####
gbm = gbm.fit(val_pred, val_label, distribution='adaboost', interaction.depth=5, n.trees=2000)
gbm_val_pred = predict(gbm.2000.sparse, data.frame(val_pred_sparse), type='response', n.trees=2000) > 0.5
accuracy_score(gbm_val_pred, val_label)
f1_score(gbm_val_pred, val_label)

rf = randomForest(df.train.cleaned[,2:ncol(df.train)], as.factor(df.train.[,1]), ntrees = 1500)
rf_val_pred = predict(rf.1500.sparse, val_pred_sparse)
accuracy_score(rf_val_pred, val_label)
f1_score(rf_val_pred, val_label)

logreg = cv.glmnet(val_pred, val_label, family = "binomial")
logreg_val_pred = predict(logreg.full, val_pred, type='response') > 0.5
accuracy_score(logreg_val_pred, val_label)
f1_score(logreg_val_pred, val_label)

logreg2 = cv.glmnet(val_pred, val_label, family = "binomial")
logreg2_val_pred = predict(logreg2.full, val_pred, type='response') > 0.5
accuracy_score(logreg2_val_pred, val_label)
f1_score(logreg2_val_pred, val_label)

nn = neuralnet(as.formula(paste('y.train~', paste(names(df.rain)[2:785], collapse='+'))), 
               data=df.train.cleaned, hidden=c(200), linear.output=FALSE)
nn_val_pred = compute(nn.200.full, val_pred)$net.result > 0.5
accuracy_score(nn_val_pred, val_label)
f1_score(nn_val_pred, val_label)

ensemble_val_pred = ((gbm_val_pred + as.numeric(rf_val_pred) + logreg_val_pred + logreg2_val_pred 
                      + nn_val_pred)/5) > 0.5
accuracy_score(ensemble_val_pred, val_label)
f1_score(ensemble_val_pred, val_label)


##### Kaggle Predictions #####
test_pred = as.matrix(binarize_data(Xtest))
test_pred_sparse = clean_features(test_pred)

gbm_test_pred = predict(gbm.2000.sparse, data.frame(test_pred_sparse), type='response', n.trees=2000) > 0.5
rf_test_pred = predict(rf.1500.sparse, test_pred_sparse)
logreg_test_pred = predict(logreg.full, test_pred, type='response') > 0.5
logreg2_test_pred = predict(logreg2.full, test_pred, type='response') > 0.5
nn_test_pred = compute(nn.200.full, test_pred)$net.result > 0.5

ensemble_test_pred = as.numeric(((gbm_test_pred + as.numeric(rf_test_pred) + logreg_test_pred 
                                  + logreg2_test_pred + nn_test_pred)/5) > 0.5)

write.table(cbind(id=1:50000, y=ensemble_test_pred), file='twitter_pred.csv', row.names=FALSE, sep=',')


##### Exploratory Data Analysis #####
data.clean = clean_rows(binarize_data(clean_features(X)), y)
X.clean = data.clean$X
y.clean = data.clean$y

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


##### Functions #####
# removes set of features from dataset of 1000 features
clean_features = function(data) {
  words <- read.csv("data/vocab.csv", header = FALSE)
  colnames(data) <- words$V1
  
  data.clean <- data[,!is.element(colnames(data),stopwords(kind="en"))]
  data.clean <- data.clean[,!is.element(colnames(data.clean),union(letters, LETTERS))]
  data.clean <- data.clean[,-grep('[[:punct:]]',colnames(data.clean))]
  data.clean <- data.clean[,-grep('[[:digit:]]',colnames(data.clean))]
  data.clean <- data.clean[,-c(14,18,84,91,150,197,213,252,268,278,334,352,403,416,474,502,587,590,624,627,677,683,729,759,790)] #manually found unicode
  colnames(data.clean) <- NULL
  
  return(data.clean)
}

# removes rows of all 0
clean_rows = function(data, labels) {
  zero.rows <- which(apply(data, 1, sum) != 0)
  data.clean <- data[zero.rows,]
  labels.clean <- labels[zero.rows]
  
  return(list(X=data.clean, y=labels.clean))
}

# normalize all counts to 0 or 1
binarize_data = function(data) {
  data.bin <- (data > 0)*1
}