v<-c(2,5.5,6)

t<-c(8,3,4)

v-t
v/t
v*t
v^t

v%%t
v%/%t
v>t
v<t
v>=t
v<=t
v==t
v!=t
x <- 0
if (x < 0) {
  print("Negative number")
} else if (x > 0) {
  print("Positive number")
} else
  print("Zero")
#for loop

x <- c(2,5,3,9,8,11,6)
m<-0
for (val in x) {
  if(val %% 2 == 0)  m = m+1
}
print(count)

#while loop

i <- 1
while (i < 6) {
  print(i)
  i = i+1
}

k<-1
while (k < 10) {
  print(k)
  k = k+1
}

#repeat loop

k<-1
repeat {
  print(k)
  k = k+1
  if (k==6){
    break
  }
}  
num <- 1:5
for(val in num) {
  if (val ==2) {
    next
  }
  print(val)
}
install.packages("dplyr")        
library(dplyr)        
setwd("D:/data w r")    
getwd()    
glimpse("Restaurant.csv")      
read.csv("Restaurant.csv")
data <- c("East","West","East","North","North","East","West","West","West","East","North")
class(data)
data_factor<- as.factor(data)
class(data_factor)
v<-c(1,2,3,4)
mat<- matrix(v,nrow=3,ncol=4,byrow = TRUE)
mat
vec1<-c(5,9,3)
vec2<-c(10,11,12,13,14,15)  
ar<-array(c(vec1,vec2),dim=c(3,3,2),byrow=TRUE)
ar
arra<-array(c(vec1,vec2),dim=c(3,3,3),byrow=TRUE)
arra  
arr<-array(c(vec1,vec2),dim=c(2,3,5))
arr        
emp_id <- c (1:5)
emp_name = c("Rick","Dan","Michelle","Ryan","Gary")
salary = c(623.3,515.2,611.0,729.0,843.25)
emp.data <- data.frame(emp_id, emp_name, salary)
emp.data
View(emp.data)
gender <- c("Male" , "Female" , "Female" , "Male")
status <- c("Poor" , "Improved", "Excellent" , "Poor" , "Excellent")
factor_gender <- factor(gender)
factor_gender
factor_status <- factor(status)
factor_status
is.factor(gender)
is.factor(factor_gender)
list_data <- list("Red", "Green", 21,32,11, TRUE, 51.23, 119.1)
list_data
getwd()
rest<-read.table("Restaurant.txt",header=TRUE)
read.csv("Restaurant.csv",header=TRUE)
View(rest)
write.table(rest,"my_rest.table")
install.packages("xlsx")
library("xlsx")
data<-matrix(c(1:10,21:30),5,4)
data
apply(data,1,sum)
apply(data,2,sum)
apply(data,1,mean)
apply(data,2,mean)
m<-matrix(c(1,2,3,4),2,2)
m
apply(m,1,sum)
apply(m,2,sum)
data<-list(x=1:5,y=6:10,z=11:15)
data
lapply(data,FUN = median)
sapply(data,FUN = median)
li<-list(a=c(1,1),b=c(2,2),c=c(3,3))
li
sapply(li,sum)
sapply(li,range)
lis<-list(a=c(1,2),b=c(1,2,3),c=c(1,2,3,4))
lis
sapply(lis,range)
age<-c(23,33,28,21,20,19,34)
gender<-c("m","m","m","f","f","f","m")
tapply(age,gender,mean)
library(datasets)
list<-list(a=c(1,1), b=c(2,2), c=c(3,3))
v<-vapply(list,sum,FUN.VALUE = double(1))
v
data()
View(mtcars)
class(mtcars)
summary(mtcars)
str(mtcars)
mtcars$cyl
mtcars$wt
tapply(mtcars$wt,mtcars$cyl,mean)
library("dplyr")
#select()
#filter()
#arrange()
#mutate()
#summarize()
select(mtcars,disp,mpg)
select(mtcars,mpg:hp)
data("iris")
View(iris)
select(iris,starts_with("petal"))
select(iris,ends_with("width"))
select(iris,contains("etal"))
filter(mtcars,cyl==8)
filter(mtcars,cyl<8)
filter(mtcars,cyl<8,vs==1)
View(filter(mtcars,cyl<8|vs==1))
arrange(mtcars,desc(disp),desc(mpg))
m_mtcars<-mutate(mtcars,my_cust_disp=disp/1.05)
View(m_mtcars)
summarise(group_by(mtcars, cyl), mean(disp))
summarise(group_by(mtcars, cyl), m = mean(disp), sd = sd(disp))
counts <- table(mtcars$gear) 
counts
barplot(counts, main="Car Distribution", 
        xlab="Number of Gear",ylab = "frequency")
barplot(counts, main = "Car Distribution", horiz = TRUE, names.arg = c("3 Gears", "4 Gears", "5 Gears"))
barplot(counts, main = "Car Distribution", xlab="Number of Gears", ylab = "Frequency", 
        legend= rownames(counts),col = c("red", "yellow", "green"))
barplot(counts, main = "Car Distribution", xlab="Number of Gears", ylab = "Frequency", 
        legend= rownames(counts),col = c("red", "yellow", "green"))
?barplot
counts <- table(mtcars$vs, mtcars$gear) 
counts
barplot(counts, main = "Car Distribution by Gears and VS", xlab="Number of Gears", 
        legend= rownames(counts),col = c("red", "yellow"))

barplot(counts, main = "Car Distribution by Gears and VS", xlab="Number of Gears", 
        legend= rownames(counts),col = c("red", "yellow"), beside = TRUE)

#Simple pie chart

slices <- c(10, 12,4, 16, 8)
lbls <- c("US", "UK", "Australia", "Germany", "France")
pie(slices, labels = lbls, main="Pie Chart of Countries")

#Pie chart with percetages to it

slices <- c(10, 12, 4, 16, 8) 
lbls <- c("US", "UK", "Australia", "Germany", "France")
pct <- round(slices/sum(slices)*100)
pct
lbls <- paste(lbls, pct) # add percents to labels 
lbls
lbls <- paste(lbls,"%",sep=" ") 
lbls
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Count")#with col option as "rainbow"
pie(slices,labels = lbls, col=rainbow(5),
    main="Pie Chart of Count")
pie(slices,labels = lbls, 
    main="Pie Chart of Count")#without any col option   

#3-d pie    
install.packages("ggplot2")
library(ggplot2)        
install.packages("plotrix")
library(plotrix)        
pie3D(slices,labels = lbls, explode = 0.1,main="Pie Chart of Count")     
#creating histogram
hist(mtcars$mpg)
#creating a colored histogarm
hist(mtcars$mpg, col="dark green", breaks=5)
?hist        
plot(density(mtcars$mpg),main = "kernel density miles per gallon")
polygon(density(mtcars$mpg),main = "kernel density miles per gallon", col="darkblue",border="red")   
View(mtcars)
weight <- c(2.5, 2.8, 3.2, 4.8, 5.1, 5.9, 6.8, 7.1, 7.8, 8.1)
months <- c(0,1,2,3,4,5,6,7,8,9)
plot(months, weight, type = "b", main = "Baby Weight Chart")
?plot 
vec <- c(3,2,5,6,4,8,1,2,3,2,4)
summary(vec)
boxplot(vec, varwidth = TRUE)
boxplot(mpg~cyl,data=mtcars, main="Car Milage Data", 
        xlab="Number of Cylinders", ylab="Miles Per Gallon")
install.packages("lsa")
library(lsa)
a <- c(-0.012813841, -0.024518383, -0.002765056,  0.079496744,  0.063928973,
       0.476156960,  0.122111977,  0.322930189,  0.400701256,  0.454048860,
       0.525526219)
b <- c(0.64175768,  0.54625694,  0.40728261,  0.24819750,  0.09406221, 
       0.16681692, -0.04211932, -0.07130129, -0.08182200, -0.08266852,
       -0.0721)
cosine_sim <- cosine(a,b)
cosine_sim
?cosine
pnorm(84,mean=72,sd=15.2,lower.tail = FALSE)

qt(c(0.025,.975),df=5)

install.packages("MASS")
library(MASS)       
head(survey)        
tbl<- table(survey$Exer,survey$Smoke)
tbl
chisq.test(tbl)
setwd("D:/data w r")
read.csv("Restaurant.csv")
stack_rest<-stack(read.csv("Restaurant.csv"))
stack_rest
names(stack_rest)<- c("sales","menu")
head(stack_rest)
avl<-aov(sales~menu,data=stack_rest)
avl
summary(avl)
#regression analysis
input<-mtcars[,c("mpg","hp","wt","disp")]
str(input)
model<-lm(mpg~hp+disp+wt,data=input)
model
head(state.x77)
class(state.x77)
st=as.data.frame(state.x77)
st
class(st)
head(st)
colnames(st)[4]<-"Life.exp"
colnames(st)[6]<-"Hs.grad"
model<- lm(Life.exp~.,data=st)
model
model1<-lm(Life.exp~Murder+Hs.grad,data=st)
model1
summary(model1)
setwd("D:/data w r")
getwd()
var<-read.csv("dataset_iceCreamConsumption.csv")
View(var)
head(var)
class(var)
Priceinc<-var$PRICE*var$INC
head(var)
interaction_model<-lm(CONSUME~PRICE+INC+TEMP+Priceinc,data=var)
interaction_model
summary(interaction_model)
install.packages("rpart")
install.packages("rpart.plot")
install.packages("RColorBrewer")
library(rpart)
library(rpart.plot)
library(RColorBrewer)
getwd()
setwd("D:/data w r")
data<-read.csv("CTG.csv")
data
str(data)
head(data)
data$NSPF<-factor(data$NSP)
str(data)
head(data)
set.seed(1234)
pd<-sample(2,nrow(data),replace=TRUE,prob=c(0.8,0.2))
train<-data[pd==1,]
validate<-data[pd==2,]
install.packages("party")
library(party)
tree<-ctree(NSPF~LB+AC+FM,data=train,controls = ctree_control(mincriterion = 0.99, minsplit = 500))
tree
plot(tree)
predict(tree,validate,type="prob")
tab <- table(predict(tree), train$NSPF)
tab
1- sum(diag(tab))/sum(tab)
testPred <- predict(tree, newdata = validate)
tab <- table(testPred, validate$NSPF)
tab
1- sum(diag(tab))/sum(tab)
install.packages("e1071")                      
library(e1071)  
install.packages("mlbench")
library(mlbench)
data(HouseVotes84, package="mlbench") 
head(HouseVotes84)
model <- naiveBayes(Class ~ ., data = HouseVotes84)
predict(model, HouseVotes84[1:20,-1])
predict(model, HouseVotes84[1:20,-1],type="raw")
pred<-predict(model,HouseVotes84[,-1])

tab <- table(pred, HouseVotes84$Class) 
tab
sum(tab[row(tab)==col(tab)])/sum(tab)
############################################################################################
getwd()
prc<-read.csv("Prostate_Cancer.csv",stringsAsFactors = FALSE)
head(prc)
prc <- prc[-1]
head(prc)
dim(prc)
table(prc$diagnosis_result)
prc$diagnosis <- factor(prc$diagnosis_result, levels = c("B", "M"), labels = c("Benign", "Malignant"))
View(prc)
str(prc)
round(prop.table(table(prc$diagnosis)) *100, digits = 1) 
?round
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
prc_n <- as.data.frame(lapply(prc[2:9], normalize))
summary(prc_n$radius)
summary(prc_n$perimeter)
summary(prc_n$area)                    
prc_train <- prc_n[1:70,]
prc_test <- prc_n[71:100,] 
prc_train_labels <- prc[1:70, 1]
prc_test_labels <- prc[71:100, 1]
library(class)
prc_test_pred <- knn(train = prc_train, test = prc_test,cl = prc_train_labels, k=8)
install.packages("gmodels")
library(gmodels)
CrossTable(x=prc_test_labels, y = prc_test_pred)
###########################################################################################
getwd()
setwd("D:/data w r")
library(e1071)
head(iris)
x = iris[,-5]
y = iris$Species
head(x)
str(x)
head(y)
str(y)
svm_model1 <- svm(x,y)
summary(svm_model1)
pred <- predict(svm_model1,x)
table(pred,y)
CrossTable(pred, y)
###############################################################################################
#clustering # kmeans clustering
setwd("D:/data w r")
cust_data<-read.csv("Insurance_Dataset_Clustering_Analysis.csv")
cust_data<-cust_data[,c(2,4,5,7,9,10,13)]
colnames(cust_data)
head(cust_data)
str(cust_data)
install.packages("amap")
library(amap)
k1<-Kmeans(cust_data,3,nstart=1,method = c("euclidean"))
k1$centers
k1$cluster
k1$size
?Kmeans
mydata <- data.frame(cust_data, k1$cluster)
head(mydata)
write.csv(mydata,"mydata.csv")     
#############################Hirarichal clustering###################################################################
colnames(mtcars)
dim(mtcars)
car.dist<-dist(mtcars)
car.dist
as.matrix(car.dist)
clusters<-hclust(car.dist)
summary(clusters)
plot(clusters)
plot(rect.hclust(clusters,h=200))
?rect.hclust
##############################agnes clustering#######################################################
library(cluster)
cluster.agnes<-agnes(mtcars)
plot(cluster.agnes)
#############################DBSCAN CLUSTERING#########################################################################
data("ruspini")
View(ruspini)
ruspini_scaled<-scale(ruspini)
plot(ruspini_scaled)
install.packages("dbscan")
library(dbscan)
kNNdistplot(ruspini_scaled,k=3)
abline(h=.25,col='red')
db<-dbscan(ruspini_scaled,eps = .25,minPts = 3)
db
plot(ruspini_scaled,col=db$cluster+1L)
#####################alternate vizualization from dbscan package##########################
hullplot(ruspini_scaled,db)
############################################Association rules###################################
install.packages("arules")
library(arules)
install.packages("arulesViz")
library(arulesViz)
data("Groceries")
head(Groceries)
View(Groceries)
str(Groceries)
rules<-apriori(Groceries,parameter = list(supp =0.001,conf = 0.8))
summary(rules)
rules<-apriori(Groceries,parameter = list(maxlen=3,supp =0.001,conf = 0.8))
summary(rules)
inspect(rules)
summary(Groceries)
rules<-apriori(Groceries,parameter = list(maxlen=3,supp =0.001,conf = 0.8), 
               appearance = list(rhs= "whole milk",default="lhs"))
summary(rules)
inspect(rules)
mydata <- read.csv('cosmetics.csv',header=TRUE,colClasses = "factor" )
head(mydata)
summary(mydata)
rules<-apriori(mydata)
summary(rules)
rules<-apriori(mydata,parameter = list(minlen=2,maxlen=3, supp=.7))
summary(rules)
rules<-apriori(mydata,parameter = list(minlen=2,maxlen=3, conf=.7),
               appearance = list(rhs=c("Foundation=Yes"),default="lhs"))
summary(rules)
plot(rules)
###############################################################################################################
#heatmap
library(ggplot2)
chicagomvt<-read.csv("motor_vehicle_theft.csv",stringsAsFactors = FALSE)
head(chicagomvt)
View(chicagomvt)
summary(chicagomvt)
chicagomvt$Date <- strptime(chicagomvt$Date, format = '%m/%d/%Y %I:%M:%S %p')
chicagomvt$Day<-weekdays(chicagomvt$Date)
chicagomvt$Hours<-chicagomvt$Date$hour
View(chicagomvt)
dailycrime<-as.data.frame(table(chicagomvt$Day,chicagomvt$Hours))
dailycrime
names(dailycrime)<-c('Day','Hours','Freq')
View(dailycrime)
dailycrime$Hours<-as.numeric(as.character(dailycrime$Hours))
dailycrime$Day<-factor(dailycrime$Day,ordered=TRUE,
                       levels = c("sunday","monday","Tuesday","Wednessday","Thursday","friday","Saturday"))
summary(dailycrime)
str(dailycrime)
ggplot(dailycrime,aes(x=Hours,y=Freq))+geom_line(aes(group=Day, color=Day))+xlab("Hour")+ylab("No of Theft")+ggtitle("daily no of motor vehicle theft")
ggplot(dailycrime, aes(x = Hours, y = Day)) + geom_tile(aes(fill = Freq)) + scale_fill_gradient(name = 'Total Motor Vehicle Thefts', low = 'white', high = 'red') + theme(axis.title.y = element_blank())
######################################################################################################
#wordcloud
slam_url <- "http://cran.r-project.org/src/contrib/Archive/slam/slam_0.1-37.tar.gz"
install.packages(slam_url, repos = NULL, type = "source")
install.packages("wordcloud")
library(wordcloud)
install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)
jeopQ <- read.csv('JEOPARDY_CSV.csv', stringsAsFactors = FALSE)
summary(jeopQ)
str(jeopQ)
jeopCorpus <- Corpus(VectorSource(jeopQ$Question))
jeopCorpus <- tm_map(jeopCorpus, content_transformer(tolower))
jeopCorpus <- tm_map(jeopCorpus, removePunctuation)
jeopCorpus <- tm_map(jeopCorpus, removeWords, stopwords('english'))
jeopCorpus <- tm_map(jeopCorpus, stemDocument)
wordcloud(jeopCorpus, max.words = 100, random.order = FALSE)