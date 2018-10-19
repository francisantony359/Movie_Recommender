library(glmnet)
library(tree)
library(class)
library(dplyr)
library(plyr)
library(mlbench)
library(caret)
library(randomForest)
library("stringr")
library(data.table)
detach("package:dplyr",unload=TRUE)
library(proxy)
library(reshape2)
library(recommenderlab)

#Reading Data

movies <- read.csv('movies.csv')
ratings<-read.csv("ratings.csv", header=T, na.strings="")
tags<-read.csv("tags.csv", header=T, na.strings="")

#Create ratings matrix. Rows = userId, Columns = movieId
ratingmat <- dcast(ratings, userId~movieId, value.var = "rating", na.rm=FALSE)
ratingmat <- as.matrix(ratingmat[,-1]) #remove userIds


#Convert rating matrix into a recommenderlab sparse matrix
ratingmat <- as(ratingmat, "realRatingMatrix")

#Normalize the data
ratingmat_norm <- normalize(ratingmat)


#Evaluation

evaluation_scheme <- evaluationScheme(ratingmat, method="cross-validation", k=5, given=3, goodRating=5) #k=5 meaning a 5-fold cross validation. given=3 meaning a Given-3 protocol
e <- evaluationScheme(ratingmat, method="split", train=0.9, 
                      k=1, given=15)
train<-getData(e, "train")
test<-getData(e, "unknown")
ratingmat_norm <- normalize(train)


algorithms <- list(
  RANDOM = list(name = "RANDOM", param = NULL),
  POPULAR = list(name = "POPULAR", param = NULL),
  UBCF1    =list(name="UBCF", param=list(method="Cosine",nn=5)),
  UBCF2     =list(name="UBCF", param=list(method="Cosine",nn=10)),
  UBCF3     =list(name="UBCF", param=list(method="Cosine",nn=20)),
  UBCF4     =list(name="UBCF", param=list(method="pearson",nn=20)),
  SVD      = list(name="SVD", param=list(k = 50))
  
)

evlist <- evaluate(evaluation_scheme, algorithms, n=c(1,3,5,10,15,20,25))
plot(evlist, legend="topleft")



#Model 
recommender_model2 <- Recommender(ratingmat_norm, method = "Popular")
recom <- predict(recommender_model2, test[3], n=12) #Obtain top 10 recommendations for 1st user in dataset
recom_list <- as(recom, "list") #convert recommenderlab object to readable list

#Obtain recommendations
recom_result <- matrix(0,10)
for (i in c(1:10)){
  recom_result[i] <- movies[as.integer(recom_list[[1]][i]),2]
}
movies[recom_result,]



#Hybrid Model

recom <- HybridRecommender(
  Recommender(train, method = "POPULAR"),
  Recommender(train, method = "UBCF"),
  Recommender(train, method = "RERECOMMEND"),
  weights = c(.6, .1, .3)
)

recom <- predict(recom, test[11], n=12) #Obtain top 10 recommendations for 1st user in dataset
recom_list <- as(recom, "list") #convert recommenderlab object to readable list

#Obtain recommendations
recom_result <- matrix(0,10)
for (i in c(1:10)){
  recom_result[i] <- movies[as.integer(recom_list[[1]][i]),2]
}
movies[recom_result,]
