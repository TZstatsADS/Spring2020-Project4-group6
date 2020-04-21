library(dplyr)
library(tidyr)
library(ggplot2)
library(tidyverse)
library(MASS)

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
# Load Data
data <- read.csv("../data/ml-latest-small/ratings.csv")
set.seed(0)
test_idx <- sample(1:nrow(data), round(nrow(data)/5, 0))
train_idx <- setdiff(1:nrow(data), test_idx)
data_train <- data[train_idx,]
data_test <- data[test_idx,]

## ========= Function: calculate RMSE ===========
# Function: calculate RMSE
RMSE <- function(rating, est_rating){
  sqr_err <- function(obs){
    sqr_error <- (obs[3] - est_rating[as.character(obs[2]), as.character(obs[1])])^2
    return(sqr_error)
  }
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}

## ========= A2: Probabilistic Matrix Factorization ===========
train = data_train
test = data_test
f = 10
lrate = 0.01
max.iter = 10
stopping.deriv = 0.01

set.seed(0)
# random assign value to matrix p and q
U = length(data$userId %>% unique())
I = length(data$movieId %>% unique())
p <- matrix(runif(f*U, -1, 1), ncol = U) 
colnames(p) <- as.character(1:U)
q <- matrix(runif(f*I, -1, 1), ncol = I)
colnames(q) <- levels(as.factor(data$movieId))

# lambda
lambda_u <- 1 / mean(apply(train, 1, var))
lambda_v <- 1 / mean(apply(train, 2, var))

for(l in 1:max.iter){
  sample_idx <- sample(1:nrow(train), nrow(train))
  for (s in sample_idx){
    # u:userid & i:movieid & r:rating
    u <- as.character(train[s,1])
    i <- as.character(train[s,2])
    r_ui <- train[s,3]
    
    # prediction error
    e_ui <- r_ui - t(q[,i]) %*% p[,u]
    
    # tune parameters
    #lambda_v <- 1 / mean(apply(train, 2, var))
    grad_q <- e_ui %*% p[,u] - lambda_v * q[,i]
    if (all(abs(grad_q) > stopping.deriv, na.rm = T)){
      q[,i] <- q[,i] + lrate * grad_q
    }
    
    #lambda_u <- 1 / mean(apply(train, 1, var))
    grad_p <- e_ui %*% q[,i] - lambda_u * p[,u]
    if (all(abs(grad_p) > stopping.deriv, na.rm = T)){
      p[,u] <- p[,u] + lrate * grad_p
    }
  }
}

est_rating <- t(q) %*% p
rownames(est_rating) <- levels(as.factor(data$movieId))
train_RMSE_cur <- RMSE(train, est_rating)
# 0.8180
train_RMSE_cur
