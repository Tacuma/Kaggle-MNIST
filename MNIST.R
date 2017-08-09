#Tacuma Solomon
#Testing Different Algorithms and Approaches for MNIST Dataset
#Description
#This Dataset is from Kaggle's famous MNIST dataset. This was basically an exercise
#in playing with different algorithms and approaches in R
#I tried:
#1.KNN
#2.C5.0 Decision tree with 10 Boosts
#3.C5.0 Decision Tree with grid search and 10 fold cross validation
#4.SVM
#5.Random Forest with 10 X 10-fold cross validation with caret grid search


library(ggplot2)
library(readr)


train <- read.csv("~/R/MINST/train.csv")
test <- read.csv("~/R/MINST/test.csv")

numTrain <- 10000
numTrees <- 25

rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows,1])
train <- train[rows,-1]


#1st Attempt -  K-Nearest Neighbors
##################################
install.packages("RWeka")
library(class)

rows <- sample(1:nrow(train), 10000)
labels <- as.factor(train[rows,1])
train <- train[rows,-1]

p <- knn(train = train, test = test, cl =  labels, k = 25)
submit <- data.frame(ImageId = 1:nrow(test), Label = p)
write.csv(submit, file = "mnistknn.csv", row.names = FALSE)


#2nd Attempt - C5.0 Decision Tree - 10 trials
#############################################
model <- C5.0(train, labels, trials = 10, costs = NULL)
p <- predict(model, test)
submit <- data.frame(ImageId = 1:nrow(test), Label = p)
write.csv(submit, file = "mnistC5.0-10trials.csv", row.names = FALSE)


#3rd Attempt - Performance tuning using caret - 10 fold cross validation and multiple trials
#############################################################################################
library(caret)
set.seed(300)
ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "oneSE")
grid <- expand.grid(.model = "tree",
                    .trials = c(10, 20, 25, 30),
                    .winnow = "FALSE")

m <- train(x = train, y = labels, method = "C5.0",
           metric = "Kappa",
           trControl = ctrl,
           tuneGrid = grid)
head(predict(m, test))
p <- predict(model, test)
submit <- data.frame(ImageId = 1:nrow(test), Label = p)
write.csv(submit, file = "mnistcaret-10foldcrossvalidation.csv", row.names = FALSE)


#4th Attempt - Performance using SVM
####################################
install.packages("kernlab")
library(kernlab)

train2 <- read.csv("~/R/MINST/train.csv")
rows2 <- sample(1:nrow(train2), 10000)
train2 <- train2[rows2,]
train2$label <- as.factor(train2$label)

model <- ksvm(label ~ ., data = train2, scaled = FALSE, kernel ="rbfdot")
p <- predict(model, test)
submit <- data.frame(ImageId = 1:nrow(test), Label = p)
write.csv(submit, file = "mnistsvm.csv", row.names = FALSE)


#5th Attempt - Performance 10 X 10 fold cv randomForest and tuning with caret
#############################################################################
set.seed(0)
library(caret)
library(randomForest)

train2 <- read.csv("~/R/MINST/train.csv")
rows2 <- sample(1:nrow(train2), 10000)
train2 <- train2[rows2,]
train2$label <- as.factor(train2$label)

ctrl <- trainControl(method = "repeatedcv",
                     number = 10, repeats = 10)
grid_rf <- expand.grid(.mtry = c(125, 250, 500))
model_rf <- train(label ~ ., data = train2, method = "rf",
                  metric = "Kappa", trControl = ctrl,
                  tuneGrid = grid_rf)


