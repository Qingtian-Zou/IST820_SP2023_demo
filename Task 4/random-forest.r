library(caTools)
library(caret)
library(randomForest)
library(R.utils)

prettySeq <- function(x) paste("Resample", gsub(" ", "0", format(seq(along = x))), sep = "")

createRandomDataPartition <- function(y, times, p) {
  vec <- 1:length(y)
  n_samples <- round(p * length(y))
  
  result <- list()
  for(t in 1:times){
    indices <- sample(vec, n_samples, replace = FALSE)
    result[[t]] <- indices
    #names(result)[t] <- paste0("Resample", t)
  }
  names(result) <- prettySeq(result)
  result
}

customSummary <- function (data,lev = NULL,model = NULL)
{
  cm <- table(factor(data$obs,c(0,1)) , factor(data$pred,c(0,1)))
  acc=(cm[1,1]+cm[2,2])/sum(cm)
  f1=cm[2,2]/(cm[2,2]+0.5*(cm[1,2]+cm[2,1]))
  dr=cm[2,2]/sum(cm[2,])
  fpr=cm[1,2]/sum(cm[1,])
  out=c(acc,f1,dr,fpr)
  names(out) <- c("Acc","F1","DR","FPR")
  out
}

set.seed(123)
seeds <- vector(mode = "list", length = 5)
for(i in 1:4) seeds[[i]] <- sample.int(1000, 36)
seeds[[5]] <- sample.int(1000, 1)

# open log file
k=0
sink(file = paste("log-",k,".txt",sep = ""),type = "output")

# load pca data
training_set=read.csv(paste("../Data/pca_training-set.csv",sep=""))
training_set$Y=factor(training_set$Y,c(0,1))
test_set=read.csv(paste("../Data/pca_test-set.csv",sep=""))
test_set$Y=factor(test_set$Y,c(0,1))

indices= createRandomDataPartition(training_set$Y, times = 4,p=0.75)
trCtrl=trainControl(method="cv",number=4,summaryFunction = customSummary, classProbs = FALSE,index = indices,seeds = seeds)

train_RandomForest=TRUE
print("############################################################")
if(train_RandomForest)
{
  rfGrid <- expand.grid(mtry = c(1:4)*ncol(training_set)%/%20)
  withTimeout(modelP.rf<- train(training_set[2:(ncol(training_set)%/%5)],training_set$Y,method="rf",metric = "F1",tuneGrid=rfGrid,trControl = trCtrl),timeout = 1200,onTimeout = "error")
  
  print("validation set results:")
  print(modelP.rf$results)
  print(modelP.rf$bestTune)
  print("training set results:")
  pred_train=predict(modelP.rf,training_set)
  print(customSummary(data.frame(obs=training_set$Y,pred=pred_train)))
  print("test set results:")
  pred_test=predict(modelP.rf,test_set)
  print(customSummary(data.frame(obs=test_set$Y,pred=pred_test)))
  
  save(modelP.rf,file= paste("RF-",k,".Rdata",sep = ""))
}
print("Random forest mode done!")
