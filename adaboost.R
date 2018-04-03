library(reshape2)
library(ggplot2)

train_test <- function() {
      train.3 <- read.csv("~/Downloads/train_3.txt", header=FALSE)
      train.8 <- read.csv("~/Downloads/train_8.txt", header=FALSE)
      
      train.3$target <- -1
      train.8$target <- 1
      p = ncol(train.3)
      
      training <- rbind(train.3, train.8)
      training <- training[, c("target", paste("V", 1:(p-1), sep = ""))]
      colnames(training) <- paste("V", 1:p, sep = "")
      training$V1 <- as.numeric(training$V1)
      
      testing <- read.table("~/Downloads/zip_test.txt", quote="\"", comment.char="")
      testing <- testing[testing$V1 %in% c(3, 8),]
      testing$V1[testing$V1 == 3] = -1
      testing$V1[testing$V1 == 8] = 1
      testing$V1 <- as.numeric(testing$V1)
      
      return(list(training, testing))
}

train <- function(X, w, y) {
      p <- ncol(X)
      n <- nrow(X)
      
      min_err <- c()
      min_thet <- c()
      min_m <- c()
      
      for (j in 1:p) {
            x_j = X[,j]
            err_thet = c()
            thetas = unique(x_j)
            t = rep(NA, length(x_j))
            
            m.ls = c()
            
            for (th in 1:length(thetas)) {
                  t[x_j>thetas[th]] = 1
                  t[x_j<=thetas[th]] = -1
                  
                  m = ifelse(mean(t==y)>0.5, 1, -1)
                  err_thet[th] = sum(w*(y != t*m))/sum(w)
                  m.ls[th] = m
            }
            
            min_index = which.min(err_thet)
            
            min_err[j] = min(err_thet)
            min_thet[j] = thetas[min_index]
            min_m[j] = m.ls[which.min(m.ls)]
      }
      
      op.j = which.min(min_err)
      op.thet = min_thet[op.j]
      op.m = min_m[op.j]
      
      return(c(op.j, op.thet, op.m))
}

classify <- function(X, pars) {
      j = pars[1]
      theta = pars[2]
      m = pars[3]
      
      x_j = X[,j]
      t = rep(NA, nrow(X))
      t[x_j>theta] = m
      t[x_j<=theta] = -m
      
      return(t)
}

agg_class <- function(X, alpha, allPars) {
      alpha = matrix(alpha)
      c_b = sapply(allPars,function(pars) classify(X, pars))
      c_hat = c_b%*%alpha
      
      c_hat[c_hat>=0] = 1
      c_hat[c_hat<0] = -1
      return(c_hat)
}

adaBoost <- function(X, y, B) {
      e.b = NA
      allPars = list()
      alpha = c()
      n = nrow(X)
      
      w = rep(1/n, n)
      
      for(b in 1:B) {
            par = train(X, w, y)
            c.b = classify(X, par)
            
            allPars[[b]] = par
            e.b = sum(w*(y != c.b))/sum(w)
            
            alpha.b = log((1-e.b)/e.b)
            alpha[[b]] = alpha.b
            w <- w*exp(alpha.b*(y != c.b))
      }   
      return(list(alpha = alpha, allPars = allPars))
}

run.adaboost <- function(training, testing, K, B) {
      train.shuffle <- training[sample(1:nrow(training)),]
      cv.index = sort(rep(1:K, len = nrow(train.shuffle)))
      
      cv.err = matrix(NA, nrow = K, ncol = B)
      test.err = matrix(NA, nrow = K, ncol = B)
      train.err = matrix(NA, nrow = K, ncol = B)
      
      test.X = testing[,-1]
      test.y = testing[, 1]
      
      for (k in 1:K) {
            train.df = train.shuffle[cv.index!=k,]
            train.X = train.df[,-1]
            train.y = train.df[,1]
            
            cv.df = train.shuffle[cv.index==k,]
            cv.X = cv.df[,-1]
            cv.y = cv.df[,1]
            
            for (b in 1:B) {
                  train.adaBoost <- adaBoost(train.X, train.y, b)
                  
                  train.pred <- agg_class(train.X, train.adaBoost$alpha, train.adaBoost$allPars)
                  train.err[k,b] = mean(train.pred !=train.y)
                  
                  cv.pred <- agg_class(cv.X, train.adaBoost$alpha, train.adaBoost$allPars)
                  cv.err[k, b] = mean(cv.pred !=cv.y)
                  
                  test.pred <- agg_class(test.X, train.adaBoost$alpha, train.adaBoost$allPars)
                  test.err[k, b] = mean(test.pred !=test.y)
            }
      }
      return(list(cv.err, test.err))
}

train_test_data = train_test()
training = train_test_data[[1]]
testing = train_test_data[[2]]

K = 5
B = 40

cv_test.err = run.adaboost(training, testing, K, B)

train.err = cv_test.err[[1]]
cv.err = cv_test.err[[2]]
test.err = cv_test.err[[3]]

train.err.mean = apply(train.err, 2, mean)
cv.err.mean = apply(cv.err, 2, mean)
test.err.mean = apply(test.err, 2, mean)

results= data.frame(Train = train.err.mean, CV = cv.err.mean, Test = test.err.mean)

melt.results = melt(results)
melt.results$B = rep(1:B, 3)
colnames(melt.results) = c("Error Type", "Error Rate", "B")
ggplot(melt.results) + 
      geom_line(aes(B,`Error Rate`, colour = `Error Type`)) 
