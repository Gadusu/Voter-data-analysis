



# Load required libraries
library(caret)
library(pROC)
library(randomForest)
library(kernlab)

# Load Voter_Data.csv dataset

dataset <- read_csv("Voter_Data_3.csv")

#Data Exploration
str(dataset)
summary(dataset)
head(dataset)
ggplot(dataset,aes(x = Age))+
  geom_histogram(binwidth = 2.0,fill = 'maroon', color ='black')+
  labs(title = 'Histogram of Age',x= "Age",y="frequency" )+
  theme(plot.title = element_text(hjust = 0.5))

dataset = dataset[6:8]

dataset$Voted_Last_Election <- factor(dataset$Voted_Last_Election, levels = c(0, 1))

# Split the data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(dataset$Voted_Last_Election, p = 0.7, list = FALSE)
training_set <- dataset[trainIndex, ]
test_set <- dataset[-trainIndex, ]




#feature scaling

preprocessParam <- preProcess(training_set ,method = c("center","scale"))
scaled_training_set <- predict(preprocessParam , training_set)
scaled_test_set <- predict(preprocessParam , test_set)

# Logistic Regression
log_reg_model <- train(Voted_Last_Election ~ Age + Salary, data = scaled_training_set, method = "glm", family = "binomial")
log_reg_pred <- predict(log_reg_model, newdata =scaled_test_set [, c("Age", "Salary")])
log_reg_cm <- confusionMatrix(log_reg_pred, scaled_test_set$Voted_Last_Election, positive = "1")
log_reg_accuracy <- log_reg_cm$overall["Accuracy"]
log_reg_precision <- log_reg_cm$byClass["Precision"]
log_reg_recall <- log_reg_cm$byClass["Recall"]
log_reg_f1 <- log_reg_cm$byClass["F1"]
log_reg_roc <- roc(as.numeric(as.character(scaled_test_set$Voted_Last_Election)), 
                   as.numeric(as.character(log_reg_pred)))
log_reg_auc <- auc(log_reg_roc)

# SVM
svm_model <- train(Voted_Last_Election ~ Age + Salary, data = scaled_training_set, method = "svmRadial")
svm_pred <- predict(svm_model, newdata = scaled_test_set[, c("Age", "Salary")])
svm_cm <- confusionMatrix(svm_pred, scaled_test_set$Voted_Last_Election, positive = "1")
svm_accuracy <- svm_cm$overall["Accuracy"]
svm_precision <- svm_cm$byClass["Precision"]
svm_recall <- svm_cm$byClass["Recall"]
svm_f1 <- svm_cm$byClass["F1"]
svm_roc <- roc(as.numeric(as.character(scaled_test_set$Voted_Last_Election)), 
               as.numeric(as.character(svm_pred)))
svm_auc <- auc(svm_roc)

# Find the optimal number of neighbors
folds <- createFolds(training_set$Voted_Last_Election, k = 10)
grid <- expand.grid(k = seq(1, 20))
ctrl <- trainControl(method = "cv", index = folds)
knn_model <- train(Voted_Last_Election ~ Age + Salary, data = training_set, method = "knn", tuneGrid = grid, trControl = ctrl)
k_opt <- knn_model$bestTune$k

# KNN
knn_model <- train(Voted_Last_Election ~ Age + Salary, data = scaled_training_set, method = "knn", tuneGrid = data.frame(k = k_opt))
knn_pred <- predict(knn_model, newdata = scaled_test_set[, c("Age", "Salary")])
knn_cm <- confusionMatrix(knn_pred, scaled_test_set$Voted_Last_Election, positive = "1")
knn_accuracy <- knn_cm$overall["Accuracy"]
knn_precision <- knn_cm$byClass["Precision"]
knn_recall <- knn_cm$byClass["Recall"]
knn_f1 <- knn_cm$byClass["F1"]
knn_roc <- roc(as.numeric(as.character(scaled_test_set$Voted_Last_Election)), 
               as.numeric(as.character(knn_pred)))
knn_auc <- auc(knn_roc)


# Decision Tree
dt_model <- train(Voted_Last_Election ~ Age + Salary, data =scaled_training_set, method = "rpart")
dt_pred <- predict(dt_model, newdata =scaled_test_set[, c("Age", "Salary")])
dt_cm <- confusionMatrix(dt_pred, scaled_test_set$Voted_Last_Election, positive = "1")
dt_accuracy <- dt_cm$overall["Accuracy"]
dt_precision <- dt_cm$byClass["Precision"]
dt_recall <- dt_cm$byClass["Recall"]
dt_f1 <- dt_cm$byClass["F1"]
dt_roc <- roc(as.numeric(as.character(scaled_test_set$Voted_Last_Election)), 
              as.numeric(as.character(dt_pred)))
dt_auc <- auc(dt_roc)

# Random Forest
rf_model <- randomForest(Voted_Last_Election ~ Age + Salary, data = scaled_training_set)
rf_pred <- predict(rf_model, newdata = scaled_test_set[, c("Age", "Salary")])
rf_cm <- confusionMatrix(rf_pred, scaled_test_set$Voted_Last_Election, positive = "1")
rf_accuracy <- rf_cm$overall["Accuracy"]
rf_precision <- rf_cm$byClass["Precision"]
rf_recall <- rf_cm$byClass["Recall"]
rf_f1 <- rf_cm$byClass["F1"]
rf_roc <- roc(as.numeric(as.character(scaled_test_set$Voted_Last_Election)), 
              as.numeric(as.character(rf_pred)))
rf_auc <- auc(rf_roc)

# Compare model performance
models <- c("Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest")
accuracy <- c(log_reg_accuracy, svm_accuracy, knn_accuracy, dt_accuracy, rf_accuracy)
precision <- c(log_reg_precision, svm_precision, knn_precision, dt_precision, rf_precision)
recall <- c(log_reg_recall, svm_recall, knn_recall, dt_recall, rf_recall)
f1 <- c(log_reg_f1, svm_f1, knn_f1, dt_f1, rf_f1)
roc <- c(log_reg_auc, svm_auc, knn_auc, dt_auc, rf_auc)

model_performance <- data.frame(models, accuracy, precision, recall, f1, roc)
model_performance

# Visualization of model performance
library(ggplot2)

# Create a bar plot for model performance
ggplot(model_performance, aes(x = models, y = accuracy, fill = models)) +
  geom_bar(stat = "identity") +
  labs(x = "Model", y = "Accuracy", title = "Model Performance") +
  theme_bw()

