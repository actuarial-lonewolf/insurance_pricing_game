# Here you have to list, following the example for caret and rpart, all the packages you want to install. 

r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

install_packages <- function() {
  install.packages("caret")
  install.packages("caretEnsemble")
  install.packages("xgboost")
  install.packages("glmnet")
  install.packages("ranger")
  install.packages("e1071")
  #install.packages("gbm")
  install.packages("rpart")
  install.packages("dplyr")
  #install.packages("tidyverse")
  #install.packages("doBy")
  #install.packages("patchwork")
  #install.packages("hrbrthemes")
  install.packages("doParallel")
  install.packages("statmod")
  install.packages("tweedie")
  install.packages("tidyr")
  #install.packages("cowplot")
  #install.packages("randomForest")
  #install.packages("Metrics")
  #install.packages("skimr")
  install.packages("tibble")
  install.packages("tictoc")
  install.packages("Metrics")
  install.packages("skimr")
  install.packages("lightgbm")
  install.packages('devtools') #in order to install catboost below
  install.packages("catboost")  #ah, needed to use the Linux version... doh!

  BINARY_URL='https://github.com/catboost/catboost/releases/download/v0.24.1/catboost-R-Linux-0.24.1.tgz'
  devtools::install_url(BINARY_URL,args = c("--no-multiarch"))

}

install_packages()

