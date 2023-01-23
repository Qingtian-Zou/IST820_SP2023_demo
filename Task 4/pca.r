library(e1071)
library(caTools)
library(R.utils)

k=0

window_size=6
window_step=1

explain=c()

# load flattened data
X=read.csv("../Data/flattened-X.csv")
X_cols=c()

# formulate column names
for(i in 1:length(names(X)))
{
  X_cols=c(X_cols,paste("X",i,sep=""))
}
Y=read.csv("../Data/flattened-Y.csv")
names(Y)=c("YX","Y")

# combine X and Y
data=data.frame(X,factor(Y$Y,levels = c(0,1)))
names(data)=c(X_cols,"Y")

data[1:(length(names(data))-1)]=scale(data[1:(length(names(data))-1)])
data=data[,colSums(is.na(data))<nrow(data)]

# apply PCA
N_data=nrow(data)
N_features=length(names(data))-1
pca.x=prcomp(data[1:N_features])
pca.var=pca.x$sdev^2
pve.x=pca.var/sum(pca.var)
# pdf(paste("PCA-variance-",k,".pdf",sep = ""))
plot(cumsum(pve.x),xlab="Principal Component",ylab ="Cumulative Proportion of Variance Explained" ,type='b')
# dev.off()
pca.data=data.frame(pca.x$x,data$Y)
names(pca.data)[length(names(pca.data))]="Y"

pca.fuzz=pca.data[1:N_data,]

# split training and test set
split = sample.split(pca.fuzz$Y, SplitRatio = 0.75)
training_set=subset(pca.fuzz,split==TRUE)
test_set=subset(pca.fuzz,split==FALSE)

# write training and test set to files
write.csv(training_set,"../Data/pca_training-set.csv")
write.csv(test_set,"../Data/pca_test-set.csv")
explain=c(explain,cumsum(pve.x)[ncol(training_set+2)%/%5-1])

print(sum(explain)/25)
