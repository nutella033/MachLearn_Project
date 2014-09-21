Predicting Weight Lifting Performance
========================================================



### Introduction

The goal of this analysis is to create an algorithm to predict the effectiveness of a weight lifting exercise based on fitness sensor data.

### Data

Below are the R packages that will be used for our analysis.


```r
library(caret)
```

We begin by importing the fitness sensor data. Our goal is to predict the `classe` variable using any of the other variables. After looking through the data manually, we determined that missing values are represented in one of three ways so we explicity specify them when importing the data.


```r
training <- read.csv("pml-training.csv",na.strings=c("","#DIV/0!","NA"))
```

Based on the needs of our analysis, we exclude the variables relating to time and the experiment participants.


```r
exclude <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp")
training <- training[,-which(names(training) %in% exclude)]
dim(training)
```

```
## [1] 19622   155
```

The data contains 154 different predictors so we look to reduce the dimensions of the dataset by eliminating predictors with zero variance.


```r
nsv <- nearZeroVar(training,saveMetrics=TRUE)
exclude2 <- rownames(nsv[nsv$zeroVar==TRUE,])
exclude2
```

```
## [1] "kurtosis_yaw_belt"      "skewness_yaw_belt"     
## [3] "amplitude_yaw_belt"     "kurtosis_yaw_dumbbell" 
## [5] "skewness_yaw_dumbbell"  "amplitude_yaw_dumbbell"
## [7] "kurtosis_yaw_forearm"   "skewness_yaw_forearm"  
## [9] "amplitude_yaw_forearm"
```

```r
training <- training[,-which(names(training) %in% exclude2)]
dim(training)
```

```
## [1] 19622   146
```

We also noticed observed that many of predictors consisted mostly, but not entirely, of `NA` values. The following function will calculate the proportion of a predictors values that are `NA`.


```r
na.prop <- function(x) {
    sum(is.na(x))/length(x)
}
```

We use the `na.prop` function to remove all predictors containing over 95% of `NA` values. This will reduce the dimensions of our dataset helping with the subsequent machine learning.


```r
exclude3 <- names(which(sapply(training,na.prop)>0.95))
training <- training[,-which(names(training) %in% exclude3)]
dim(training)
```

```
## [1] 19622    55
```


### Model

We now partition our cleaned up dataset to create training and testing sets for our model development.


```r
set.seed(1000)
inTraining <- createDataPartition(y=training$classe,p=0.8,list=FALSE)
myTraining <- training[inTraining,]
myTesting <- training[-inTraining,]
```

To optimize our model development we apply cross-validation with 10 folds using the `trainControl` function.


```r
fitControl <- trainControl(method="cv",number=10)
```

We now use random forests to create the prediction algorithm as it generally provides high accuracy for class prediction.


```r
modFit <- train(classe ~ .,data=myTraining,trControl=fitControl,method="rf")
```




```r
summary(modFit)
```

```
##                 Length Class      Mode     
## call                4  -none-     call     
## type                1  -none-     character
## predicted       15699  factor     numeric  
## err.rate         3000  -none-     numeric  
## confusion          30  -none-     numeric  
## votes           78495  matrix     numeric  
## oob.times       15699  -none-     numeric  
## classes             5  -none-     character
## importance         54  -none-     numeric  
## importanceSD        0  -none-     NULL     
## localImportance     0  -none-     NULL     
## proximity           0  -none-     NULL     
## ntree               1  -none-     numeric  
## mtry                1  -none-     numeric  
## forest             14  -none-     list     
## y               15699  factor     numeric  
## test                0  -none-     NULL     
## inbag               0  -none-     NULL     
## xNames             54  -none-     character
## problemType         1  -none-     character
## tuneValue           1  data.frame list     
## obsLevels           5  -none-     character
```

```r
modFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4462    1    0    0    1    0.000448
## B    7 3027    4    0    0    0.003621
## C    0    6 2732    0    0    0.002191
## D    0    0    8 2564    1    0.003498
## E    0    1    0    2 2883    0.001040
```


### Accuracy


```r
print(modFit,digits=3)
```

```
## Random Forest 
## 
## 15699 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 14130, 14127, 14130, 14127, 14130, 14131, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    0.996     0.995  0.00157      0.001981
##   28    0.998     0.998  0.00070      0.000886
##   54    0.996     0.995  0.00180      0.002276
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```

Based on the results of the model fitting, our in-sample accuracy is 0.998, making our in-sample error rate 0.002. We can now apply our model to our training set to determine the out-sample error rate.


```r
pred <- predict(modFit,myTesting)
accuracy <- sum(pred==myTesting$classe)/nrow(myTesting)
accuracy
```

```
## [1] 0.9982
```

```r
error <- 1-accuracy
error
```

```
## [1] 0.001784
```

The estimate of our out-sample accuracy is 0.9982, making our out-sample error rate 0.001, or 0.1%.
