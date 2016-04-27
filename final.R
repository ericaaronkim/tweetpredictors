library(glmnet)
library(randomForest)
library(gbm)
library(neuralnet)

setwd('~/Documents/class/stat154/final_proj/') # Set directory

load('TrainTest.RData')

x_matrix = as.matrix(X)
x_clean = x_matrix[rowSums(x_matrix) != 0,]
y_clean = y[rowSums(x_matrix) != 0]
x_binarized = matrix(as.numeric(x_clean > 0), nrow=nrow(x_clean))

data_subset_ind = sample(nrow(x_binarized), 49712)
data_subset_pred = as.matrix(x_binarized[data_subset_ind,])
data_subset_label = y_clean[data_subset_ind]

# subset training/validation data
val_samp_ind = sample(49712, 10000)
val_samp_pred = as.matrix(data_subset_pred[val_samp_ind,])
val_samp_label = data_subset_label[val_samp_ind]

train_samp_pred = as.matrix(data_subset_pred[-val_samp_ind,])
train_samp_label = data_subset_label[-val_samp_ind]

p = data.frame(train_samp_pred)
l = data.frame(train_samp_label)
d = cbind(l, p)
# train neural net
neural_net = neuralnet(as.formula(paste('train_samp_label~', paste(names(p), collapse='+'))), 
                       data=d, hidden=c(500), linear.output=FALSE)
val_pred_nn = compute(neural_net, val_samp_pred)$net.result > 0.5
sum(val_pred_nn==val_samp_label)/10000

# train random forest
random_forest = randomForest(train_samp_pred, train_samp_label, ntrees=1000)
val_pred_rf = predict(random_forest, val_samp_pred, type='response') > 0.5
sum(val_pred_rf==val_samp_label)/10000

# train logistic model
logistic_model = cv.glmnet(train_samp_pred, train_samp_label, family='binomial')
val_pred_log = predict(logistic_model, val_samp_pred, type='response') > 0.5
sum(val_pred_log==val_samp_label)/10000

# train gbm model
gbm_model = gbm.fit(data_subset_pred, data_subset_label, distribution='adaboost',
                    interaction.depth=5, n.trees=100)
val_pred_gbm = predict(gbm_model, val_samp_pred, type='response', n.trees=100) > 0.5
sum(val_pred_gbm==val_samp_label)/10000


test_pred = as.numeric(compute(neural_net, Xtest)$net.result > 0.5)
write.table(cbind(id=1:50000, y=test_pred), file='twitter_pred.csv', row.names=FALSE, sep=',')

save.image('proj.RData')
