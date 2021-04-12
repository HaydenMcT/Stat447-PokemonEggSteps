## CODE FILE 2e: Given cleaned train and holdout Pokemon datasets from Code File 01d, performs K-Nearest Neighbors on the data, and saves
## the models' assessed performance results.

############################################
###STEP 0: Loading libraries and datasets###
############################################
library(tidyverse) # allows simpler code for dataset manipulation
library(tidymodels) # makes data pre-processing and results validation easier
library(kknn) # allows performing of Weighted k-Nearest Neighbors Classification, Regression and Clustering
library(caret) # allows streamlining of model training process for complex classification

load("01d-ValidTrainSets.RData") #data
load("02a-Utils.RData") #functions


#####################################################################
###STEP 1.1: Encoding factor levels of response variable to labels###
#####################################################################

# ordered levels of base_egg_steps will be encoded to labels to preserve functionality
# of grepl in Utils.R :
# S : Short , for level "<=3840"
# M : Moderate, for level "5120"
# L : Long, for level "6400"
# E: Extreme, for level ">=7680"

train$base_egg_steps = factor((EncodeToChar(train$base_egg_steps,
                                            c("S","M","L","E"))),
                              levels=c("S","M","L","E"),
                              ordered=TRUE)
holdout$base_egg_steps = factor((EncodeToChar(holdout$base_egg_steps,
                                              c("S","M","L","E"))),
                                levels=c("S","M","L","E"),
                                ordered=TRUE)


#############################################################
###STEP 1.2: Create numerical-only datasets for KNN models###
#############################################################

#set.seed(254)
train = select_if(train, is.numeric) %>% 
  cbind("base_egg_steps" = train$base_egg_steps)

holdout = select_if(holdout, is.numeric) %>% 
  cbind("base_egg_steps" = holdout$base_egg_steps)

#View(train); View(holdout)


#############################################################
###STEP 2.1: Make functions to fit base data to KNN Model####
#############################################################

#' @description Runs Utility functions across the holdout set
#' @param holdout A data set which contains base_egg_steps and the relevant prediction
#' data
#' @param predprobs Predicted probabilities for each of the four classes
#' @param labels Defaults to single levels as encoded at the beginning of the file
#' which correspont to how many egg steps are needed
#' @param siglev The two significance levels which helper functions will use
#' Defaults to 0.5 and 0.8
#' @return 

RunUtilsKNN = function(holdout, predprobs, labels = c("S","M","L","E"),
                       siglev=c(0.5, 0.8)){
  modOrdInt = OrdinalPredInterval(predprobs, labels, 
                                  level1 = siglev[1], level2 = siglev[2])
  modTableL1 = table(holdout$base_egg_steps, modOrdInt$pred1)
  modTableL2 = table(holdout$base_egg_steps, modOrdInt$pred2)
  modIntLossL1 = predIntervalLoss(modOrdInt$pred1, 
                                  true_labels=holdout$base_egg_steps)
  modIntLossL2 = predIntervalLoss(modOrdInt$pred2, 
                                  true_labels=holdout$base_egg_steps)
  modPerformL1 = coverage_across_classes(modTableL1)
  modPerformL2 = coverage_across_classes(modTableL2)
  UtilsOut = list(modOrdInt=modOrdInt, modTableL1=modTableL1,
                  modTableL2=modTableL2, modIntLossL1=modIntLossL1,
                  modIntLossL2=modIntLossL2, modPerformL1=modPerformL1,
                  modPerformL2=modPerformL2)
  return(UtilsOut)
}


#' @description Takes a training and holdout set and finds the optimal K value for
#' a K-nearest-neighbors model using the set number of folds provided, tuning across
#' the set amount of K values.  It then outputs the accuracy, predictions, and performance
#' metric functions
#' @param train the training set for forming the model
#' @param holdout the test set for assessing the model
#' @param folds the number of folds used in tuning
#' The default is 5 folds
#' @param seed The seed value for reproducibility
#' The default is 254 (no significance to this value)
#' @param siglev The significance levels for the utility functions
#' The default values are 0.5 and 0.8
#' @param k_vals The sequence of K values that the model will be tuned over
#' The default is sequence of 1 to 30 increasing by 1
#' @details The distance used is Euclidean.  The tidyverse and tidymodels packages
#'  are REQUIRED to run this function.  base_egg_steps MUST be the response variable.
#'  The chosen engine is KKNN, and this packages is REQUIRED
#' @return A list which contains:
#'  The models performance across our metrics on the holdout set (Model)
#'  The accuracy value on the holdout set (BestKNNaccuracy)
#'  The predictions on the holdout set (BestKNNpred)
#'  The predicted probabilities for each class on the holdout set (BestKNNpredprobs)
#'  The best K value (bestK)
#'  An accuracy plot on the training sets through tuning is printed
FormKNN = function(train, holdout, folds = 5, seed = 254, siglev=c(0.5, 0.8),
                   k_vals = tibble(neighbors = seq(1, 30, 1))){
  set.seed(seed)
  knn_recipe = recipe(base_egg_steps~., data = train) %>% 
    step_scale(all_predictors()) %>%
    step_center(all_predictors())
  poke_vfold = vfold_cv(train, v = folds, strata = base_egg_steps)
  knn_spec = nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>% 
    set_engine("kknn") %>%
    set_mode("classification")
  knn_workflow = workflow() %>% 
    add_recipe(knn_recipe) %>% 
    add_model(knn_spec) %>%
    tune_grid(resamples = poke_vfold, grid = k_vals) %>% 
    collect_metrics()
  knn_accuracies = knn_workflow %>% 
    filter(.metric == "accuracy")
  auccuracy_plot = ggplot(knn_accuracies, aes(x = neighbors, y = mean)) +
    geom_point() + geom_line() +
    labs(x = "Neighbors", y = "Accuracy Estimate", title = "KNN Accuracy Plot") +
    theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 20))
  print(auccuracy_plot)
  bestK = knn_accuracies %>% 
    arrange(desc(mean)) %>% 
    slice(1) %>% 
    pull(neighbors)
  knn_BestMod = nearest_neighbor(weight_func = "rectangular", neighbors = bestK) %>% 
    set_engine("kknn") %>%
    set_mode("classification")
  BestKNNfit = workflow() %>% 
    add_recipe(knn_recipe) %>% 
    add_model(knn_BestMod) %>% 
    fit(data = train)
  BestKNNpred = BestKNNfit %>% 
    predict(holdout) %>% 
    bind_cols(holdout)
  BestKNNpredprobs = BestKNNfit %>% 
    predict(holdout, type = "prob") %>% 
    mutate(.pred_E = .pred_E-.pred_L,
           .pred_L = .pred_L-.pred_M,
           .pred_M = .pred_M-.pred_S)
  BestKNNaccuracy = metrics(BestKNNpred, truth = base_egg_steps, 
                            estimate = .pred_class)
  Model = RunUtilsKNN(holdout, BestKNNpredprobs, siglev=siglev)
  resultsList = list(Model=Model, BestKNNaccuracy=BestKNNaccuracy, 
                     BestKNNpred=BestKNNpred, BestKNNpredprobs=BestKNNpredprobs,
                     bestK=bestK)
  return(resultsList)
}

# Evaluating KNN on all of the numeric data
AllNumericKNN = FormKNN(train, holdout)


################################################
###STEP 2.2: Fit upsampled data to KNN Model####
################################################

#Upsampling is used to attempt to improve on the class imbalance
trainUpSamp = upSample(train, train$base_egg_steps) %>% 
  select(-Class)
TrainRows = sample(nrow(trainUpSamp))
trainUpSamp = trainUpSamp[TrainRows,]

# Evaluating KNN on the upsampled data
UpsampledKNN = FormKNN(trainUpSamp, holdout)


# A data frame of the performances on our assessment metrics
knn_performance = data.frame(cbind(AllNumericKNN$Model$modPerformL1, 
                              AllNumericKNN$Model$modPerformL2, 
                              UpsampledKNN$Model$modPerformL1, 
                              UpsampledKNN$Model$modPerformL2))


####################################################################################################
###STEP 3: Saving all relevant functions, objects and models, including their average performances###
#####################################################################################################

save(file="KNNmodelPerformances.RData", AllNumericKNN, UpsampledKNN, RunUtilsKNN, FormKNN, 
     knn_performance)