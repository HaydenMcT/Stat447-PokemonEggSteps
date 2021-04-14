## CODE FILE 2b: Given cleaned train and holdout Pokemon datasets from Code File 01d, fits Random Forest to the data, selects
## variables that result in the best performance. Saves the best model and its selected variables, to be used for cross validation in Phase C/
## Phase 3.

############################################
###STEP 0: Loading libraries and datasets###
############################################

library(randomForest) # allows use of random forests
library(rpart) # allows use of decision trees
library(rpart.plot) # allows plotting of decision trees
library(caret) # allows streamlining of model training process for complex classification
library(tidyverse) # allows simpler code for dataset manipulation

load("RdataFiles/ValidTrainSets.RData") #data
load("RDataFiles/Utils.RData") #functions
set.seed(0)


#####################################################################
###STEP 1.1: Encoding factor levels of response variable to labels###
#####################################################################

# base_egg_steps is converted to single-character labels to preserve functionality
# of grepl in Utils.R:
# S : Short, level <=3840
# M : Moderate, level 5120
# L : Long, level "400
# E: Extreme,  level >=7680

train$base_egg_steps = factor((EncodeToChar(train$base_egg_steps,
                              c("S","M","L","E"))),
                              levels=c("S","M","L","E"),
                              ordered=TRUE)
holdout$base_egg_steps = factor((EncodeToChar(holdout$base_egg_steps, 
                                  c("S","M","L","E"))),
                                  levels=c("S","M","L","E"),
                                  ordered=TRUE)

#View(train); View(holdout)

###########################################################################
###STEP 1.2: Creating functions in order to run and assess Random Forest###
###########################################################################

#' @description A function which creates a random forest model and then
#'  runs the utility functions for a given set of predictions and their 
#'  corresponding helper inputs
#'  @param train This is a data frame with the training set and a class of single-
#'  character class labels
#'  @param test This is a data frame with the holdout set and a class of single-
#'  character class labels
#'  @param labels a vector of all the relevant labels, defaluts to those we use here
#'  @param siglev a vector of two significance levels
#'  @return A list containing the model, the predicted classes and probabilities
#'  all outputs from the utility functions, and tables which are confusion matricies

ForestAndUtils = function(train, test, labels, siglev){
  RFmodel = randomForest(base_egg_steps~., data = train, importance = T, proximity = T)
  RFmodoutpredprob = predict(RFmodel, newdata = test, type = "prob")
  RFmodoutpred = predict(RFmodel, newdata = test)
  predintRF =  OrdinalPredInterval(RFmodoutpredprob,
                                    labels = labels,
                                    level1=siglev[1], level2=siglev[2])
  intervalLossL1 = PredIntervalLoss(predintRF$pred1,
                                    true_labels=test$base_egg_steps)
  intervalLossL2 = PredIntervalLoss(predintRF$pred2,
                                    true_labels=test$base_egg_steps)
  TableL1 = table(test$base_egg_steps, predintRF$pred1)
  TableL2 = table(test$base_egg_steps, predintRF$pred2)
  performanceL1 = CoverageAcrossClasses(TableL1)
  performanceL2 = CoverageAcrossClasses(TableL2)
  return(UtilOut = list(RFmodel=RFmodel, RFmodoutpredprob = RFmodoutpredprob,
                        RFmodoutpred=RFmodoutpred, predintRF=predintRF,
                        intervalLossL1=intervalLossL1, intervalLossL2=intervalLossL2,
                        TableL1=TableL1, TableL2=TableL2, performanceL1=performanceL1,
                        performanceL2=performanceL2))
}


#' @description A function which, given a data set and an importance threshold,
#' produces a random forest model that has all variables above the importance
#' threshold by eliminating variables using backwards step-wise regression
#' @param train A training set with base_egg_steps and the predictor variables
#' @param test The testing set with base_egg_steps and the predictor variables
#' @param GINIthreshold A level which all variables in the model must be above for
#' the backwards elimination to stop
#' @param LossThreshold A threshold for which the next model in backwards stepwise
#' model creation can not be worse by more than this value
#' @param labels used in metric functions
#' The defaults are the SMLE format
#' @param siglevels the significance levels to test at for the metric function
#' The defaults are 0.5 and 0.8
#' @param L1 A boolean indicating whether or not to to use the first significance 
#' level or the second significance level for the Loss Difference (as associated with LossThreshold)
#' The default value is TRUE, indicating the use of the first significance level
#' @details The function will continue to backwards select variables and form new
#' random forests until EITHER the GINI threshold is surpassed or the LossThreshold
#' is surpassed
#' @return A list of ALL models created up until the the repeat condition is broken.
#' Each model-specific entry in the list includes the model and all the metric functions
#' for this model
RandForstParse = function(train, test, GINIthreshold, LossThreshold,
                          labels=c("S","M","L","E"),
                          siglev=c(0.5,0.8), L1=TRUE){
  UtilOut = ForestAndUtils(train, test, labels, siglev)
  RFmod = UtilOut$RFmodel
  GINIval = min(RFmod$importance[,6])
  poorVar = names(which.min(RFmod$importance[,6]))
  UtilOutOld = UtilOut
  ModStore = list()
  ModStore[[1]] = UtilOutOld
  list_idx = 1
  if(L1){
    LossDiff = UtilOut$intervalLossL1 - UtilOutOld$intervalLossL1
  }
  else{LossDiff = UtilOut$intervalLossL2 - UtilOutOld$intervalLossL2}
  while (GINIval <= GINIthreshold && 
         LossDiff <= LossThreshold && 
         length(names(train))>=3){
    UtilOutOld = UtilOut
    train = select(train, -poorVar)#; print(names(train))
    UtilOut = ForestAndUtils(train, test, labels, siglev)
    RFmod = UtilOut$RFmodel
    GINIval = min(RFmod$importance[,6])
    poorVar = names(which.min(RFmod$importance[,6]))
    if(L1){
      LossDiff = UtilOut$intervalLossL1 - UtilOutOld$intervalLossL1
    }
    else{LossDiff = UtilOut$intervalLossL2 - UtilOutOld$intervalLossL2}
    ModStore[[list_idx]] = UtilOutOld
    list_idx = list_idx + 1
    #print(GINIval); print(poorVar); print(LossDiff)
    #print(UtilOut$RFmodoutpredprob[1:3,])
  }
  return(ModStore)
}


#' @description A model that extras the best model given the threshold of choice
#' from a RandForestParse Object
#' @param RandForestParseObj A list of all models and metrics as given by the RandForestParse
#' function
#' @param L1 A boolean indicating whether or not to use the lower or higher significance
#' level for model selection
#' The default value is TRUE, which selects the first (lower) significance level
#' @details Scans all objects in the list to find the lowest Prediction Interval 
#' Loss value
#' @return The best model object.  This object includes the model itself and all
#' of the mectirc functions we are using in the form of a list
BestMods = function(RandForestParseObj, L1 = TRUE){
  best_loss = 1
  best_model = RandForestParseObj[[1]]
  for (UtilObj in RandForestParseObj){
    if (L1) {curr_loss = UtilObj$intervalLossL1}
    else {curr_loss = UtilObj$intervalLossL2}
    if (curr_loss < best_loss){
      best_loss = curr_loss
      best_model = UtilObj
    }
  }
  return(best_model)
}


####################################################################################
###STEP 2: Creating Random Forest models using various sets of selected variables###
####################################################################################

## A model using all the variables in the dataset
full_mod_rf = ForestAndUtils(train, holdout,
                           labels=c("S","M","L","E"),
                           siglev=c(0.5,0.8))
## The best model using backwards selection on the first significance level of 0.5
## This uses a very high threshold on both metrics to go through a full run until
## reaching the null model.  We then use the best model extractor to collect the
## best model on found
find_best_iL1 = RandForstParse(train = train, test = holdout,
                             GINIthreshold = 100, LossThreshold = 1, L1 = TRUE)
best_iL1 = BestMods(find_best_iL1, L1 = TRUE)

## Same process but using the second significance level, 0.8
find_best_iL2 = RandForstParse(train = train, test = holdout,
                             GINIthreshold = 100, LossThreshold = 1, L1 = FALSE)
best_iL2 = BestMods(find_best_iL2, L1 = FALSE)

## This creates a model using upsampled data in an attempt address the class imbalance
train_upsamp = upSample(train, train$base_egg_steps) %>% 
  select(-Class)
train_rows = sample(nrow(train_upsamp))
train_upsamp = train_upsamp[train_rows,]
upsamp_full_mod = ForestAndUtils(train_upsamp, holdout, 
                               labels=c("S","M","L","E"),
                               siglev=c(0.5,0.8))

## This selects the same variables found when creating the multinomial model and
## fits a random forest model to them !!! Or was this a union with L1
multinomial_rf_mod_train = select(train, base_egg_steps, base_total, 
                          capture_rate, is_dragon_type, 
                          is_ground_type, weight_kg, is_normal_type,
                          is_bug_type, experience_growth)
multinomial_rf_mod_holdo = select(holdout, base_egg_steps, base_total, 
                               capture_rate, is_dragon_type, 
                               is_ground_type, weight_kg, is_normal_type,
                               is_bug_type, experience_growth)
multinomial_rf = ForestAndUtils(multinomial_rf_mod_train, multinomial_rf_mod_holdo,
                               labels=c("S","M","L","E"), siglev=c(0.5,0.8))

## This selects the same variables found when creating the ordinal model and
## fits a random forest model to them !!! Or was this a union with L1
ordinal_rf_train = select(train, base_egg_steps, is_rock_type, is_dragon_type, 
                       capture_rate, is_ghost_type)
ordinal_rf_holdout = select(holdout, base_egg_steps, is_rock_type, is_dragon_type, 
                       capture_rate, is_ghost_type)
ordinal_rf = ForestAndUtils(ordinal_rf_train, ordinal_rf_holdout,
                           labels=c("S","M","L","E"), siglev=c(0.5,0.8))

## Fits a model to the set of variables which forms a union from the best multinomial
## model and the best backwards RF model at 0.8
multinomial_best_80rf_train = select(train, base_egg_steps, base_total, 
                               capture_rate, is_dragon_type, 
                               is_ground_type, weight_kg, is_normal_type,
                               is_bug_type, experience_growth, sp_attack)
multinomial_best_80rf_holdo = select(holdout, base_egg_steps, base_total, 
                                 capture_rate, is_dragon_type, 
                                 is_ground_type, weight_kg, is_normal_type,
                                 is_bug_type, experience_growth, sp_attack)
multi_best_80rf = ForestAndUtils(multinomial_best_80rf_train, multinomial_best_80rf_holdo,
                               labels=c("S","M","L","E"), siglev=c(0.5,0.8))

## Fits a model to the set of variables which forms a union from the best ordinal
## model and the best backwards RF model at 0.8
ordinal_rf_train2 = select(train, base_egg_steps, is_rock_type, is_dragon_type, 
                       capture_rate, is_ghost_type, weight_kg, base_total, sp_attack)
ordinal_rf_holdout2 = select(holdout, base_egg_steps, is_rock_type, is_dragon_type, 
                         capture_rate, is_ghost_type, weight_kg, base_total, sp_attack)
ordinal_rf2 = ForestAndUtils(ordinal_rf_train2, ordinal_rf_holdout2,
                           labels=c("S","M","L","E"), siglev=c(0.5,0.8))


#########################################################
###STEP 3: Comparing performances of all our RF models###
#########################################################

## A data frame of all the performance results side-by-side to facilitate comparison
average_performances = data.frame(cbind(c(full_mod_rf$performanceL1[2], full_mod_rf$performanceL2[2], full_mod_rf$intervalLossL1, full_mod_rf$intervalLossL2),
                                        c(best_iL1$performanceL1[2], best_iL1$performanceL2[2], best_iL1$intervalLossL1, best_iL1$intervalLossL2),
                                        c(best_iL2$performanceL1[2], best_iL2$performanceL2[2], best_iL2$intervalLossL1, best_iL2$intervalLossL2),
                                        c(multi_best_80rf$performanceL1[2], multi_best_80rf$performanceL2[2], multi_best_80rf$intervalLossL1, multi_best_80rf$intervalLossL2),
                                        c(upsamp_full_mod$performanceL1[2], upsamp_full_mod$performanceL2[2], upsamp_full_mod$intervalLossL1, upsamp_full_mod$intervalLossL2)
                                  ))
rownames(average_performances) = c("Coverage(50% pred interval)", "Coverage(80% pred interval)", "Loss (50% pred interval)", "Loss (80% pred interval)")
names(average_performances) = c("All_Features", 
                                "Back_Sel(50%)",
                                "Back_Sel(80%)", 
                                "Back_Sel_and_Best_Multin",
                                "All_Features(Upsampling)")
print(average_performances)

# A list of the models which were not used as they were too poor
unused_rf_models=list(ordinal_rf, ordinal_rf2, multinomial_rf)

###################################################################
###STEP 4.1: Create functions to fit the best model to new data ###
###          and make predictions with it                       ###
###################################################################

#' @description
#' Fits previously selected RF model to the provided dataset
#' 
#' @param dataset Dataset on which to fit the model. response variable is assumed to be 
#'                the last column of the dataset, and every other column is assumed to be
#'                an explanatory variable.
#'                
#' @return a classification tree model object, fit to the provided dataset and capable of making new predictions
#' 
RFFitter = function(data, formula = base_egg_steps ~ base_total + capture_rate +
                      is_dragon_type + is_ground_type + weight_kg +
                      is_normal_type + is_bug_type + experience_growth + sp_attack) {
  return(randomForest(formula, data=data, importance = T, proximity = T))
}

#' Provides the predicted probabilities, according to a provided model, 
#' of being in each possible class for each example in a given dataset.
#' 
#' @param data Data on which to make predictions. Predicted response variable is assumed to be 
#'             binary.
#'             
#' @param model A model to use to make predictions
#'                
#' @return The predicted probabilities, for each example in data and for each possible class
#' 
RFPredictor = function(data, model) {
  return(predict(model, newdata=data, type="prob"))
}


#####################################################################################################
###STEP 4.2: Saving best model, functions to use it on new data, and relevant performance metrics ###
#####################################################################################################

best_RF_model = multi_best_80rf
RF_features = rownames(best_RF_model$RFmodel$importance)
RF_pred_interval_losses = c(best_RF_model$intervalLossL1, best_RF_model$intervalLossL2)
RF_selection_table = average_performances

save(file="RDataFiles/RandomForestModel.RData", best_RF_model, RF_features, RF_pred_interval_losses, RF_selection_table, RFFitter, RFPredictor)
