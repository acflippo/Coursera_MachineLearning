# Predicting Barbell Lifts
Annie Flippo  
October 14, 2014  

# Executive Summary
Human wearable devices have increased dramatically in the past few years.  These devices have made it possible to collect large amount of data in human activities.  In this study, the data collected in the Qualitative Activity Recognition of Weight Lifting Exercises project (see reference below) were used to categorgized barbell lifts.  The participants were asked to perform to lift in one correct and four incorrect manners.  These were categorized in the five following classes:

* Class A - exactly according to the specification
* Class B - throwing the elbows to the front 
* Class C - lifting the dumbbell only halfway  
* Class D - lowering the dumbbell only halfway 
* Class E - throwing the hips to the front 

The data was divided into 70% as a training set and 30% as a validation set.  A classification model using a Random Forest algorithm was employed to train and validate the model.  The In-Sample error rate for the training set was 0.22% and the Out-of-Sample error was 0.3%.  In this study, the Random Forest algorithm performed extremely well on classifying the 5 classes of barbell lifts and in the predictions of the test cases. 

## Get the Data

```r
library(caret); library(e1071); library(ggplot2);library(corrgram);library(doSNOW);
registerDoSNOW(makeCluster(2, type = "SOCK"))
setwd("~/Dropbox/Coursera_MachineLearning")
if (!file.exists("pml-training.csv")) {
    trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(trainURL, destfile="pml-training.csv", method="curl")
    download.file(testURL, destfile="pml-testing.csv", method="curl")
}
train.raw <- read.csv("pml-training.csv")
test.raw <- read.csv("pml-testing.csv")
```

## Data Preparation
Many columns have missing data or zero-length string.  Since missing values do not contribute to the model, they are excluded in the data set.  In addition, names of participants and dates are also eliminated.  Columns with more than 90% of missing data or zero-length string were removed from the study.  Finally, the data was partitioned 70% for training and 30% for validation as follows.


```r
set.seed(123456)
inTrain = createDataPartition(train.raw$classe, p = 0.7)[[1]]
# Find columns with mostly NAs and keep only columns with more than 90% of data
nonNAcolumn <- (apply(train.raw, 2, function(x) sum(!is.na(x)))) > (nrow(train.raw) * 0.9)

# Find columns with mostly data and keep only columns that are more than 90% non-blanks
nonBlankColumn <- (apply(train.raw, 2, function(x) sum(nchar(x) != 0))) > (nrow(train.raw) * 0.9)
columnsToKeep <- nonNAcolumn & nonBlankColumn

# Include only columns that has data
training <- train.raw [, columnsToKeep]
# Remove first 5 columns of name and dates
training <- training [, 6:ncol(training)]

train.data <- training[inTrain,]
validate.data <- training[-inTrain,]
test.data <- test.raw[, columnsToKeep]
test.data <- test.data[, 6:ncol(test.data)]
```

The final data set include 55 predictors and has 13737 rows.  There were too many predictors to generate a sensible correlation plot but a sample graph of 2 predictors is created and color-coded by classe.  As an illustration, there is some evidence of clustering for the different classes in the roll\_belt and pitch\_forearm dimension.

```r
p <- ggplot(train.data, aes(pitch_forearm, roll_belt))
p + geom_point(aes(colour=train.data$classe))
```

![plot of chunk unnamed-chunk-3](./Predict_Barbell_Lifts_Project_mac_files/figure-html/unnamed-chunk-3.png) 

## Modeling
A Random Forest algorithm is run for the training data.

```r
set.seed(123456)
system.time(
modelFit <- train(classe ~., data=train.data, method="rf", importance=TRUE,
                  trControl = trainControl(method = "cv", number = 4))
)
```

```
##    user  system elapsed 
##  93.546   0.304 512.555
```

```r
print(modelFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 0.22%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3905    0    0    0    1    0.000256
## B    7 2648    3    0    0    0.003762
## C    0    4 2391    1    0    0.002087
## D    0    0    8 2243    1    0.003996
## E    0    0    0    5 2520    0.001980
```

With the In-Sample error rate is 0.22%, I **expect the Out-of-Sample error rate to be slight higher than 0.22%**.  As the number of trees increases, the error rate decreases as illustrated by the following graph:
![plot of chunk unnamed-chunk-5](./Predict_Barbell_Lifts_Project_mac_files/figure-html/unnamed-chunk-5.png) 

A Variable Importance graph for the model per classe of barbell lifts:
![plot of chunk unnamed-chunk-6](./Predict_Barbell_Lifts_Project_mac_files/figure-html/unnamed-chunk-6.png) 

## Validating the Model
The modelFit object was used to predict the outcome classe for the validation data set aside earlier.  The accuracy for the predictions in the validation data is 0.997 which gives an error rate of 0.003 (or 0.3%).

```r
# Using Validate data set aside
predValue <- predict(modelFit, validate.data)
confusionMatrix(predValue,validate.data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    7    0    0    0
##          B    0 1132    2    0    0
##          C    0    0 1024    4    0
##          D    0    0    0  960    3
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.996, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.994    0.998    0.996    0.997
## Specificity             0.998    1.000    0.999    0.999    1.000
## Pos Pred Value          0.996    0.998    0.996    0.997    1.000
## Neg Pred Value          1.000    0.999    1.000    0.999    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.174    0.163    0.183
## Detection Prevalence    0.286    0.193    0.175    0.164    0.183
## Balanced Accuracy       0.999    0.997    0.999    0.998    0.999
```

## Prediction on test data
In the final project submission, I predicted the classe for 20 test cases. 

```r
# Predicting using test data set 
predTestValue <- predict(modelFit, test.data)
predTestValue
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion
The Random Forest model performed very well with an In-Sample error rate of 0.22% and a Out-of-Sample error rate of 0.3%.  It also predicted the test data with great accuracy. 

### References
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz3GXgmffpY
