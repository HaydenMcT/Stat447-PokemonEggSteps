## CODE FILE 2c: Given cleaned train and holdout Pokemon datasets from Code File 01d, fits Multinomial Logistic Regression to the data, selects
## variables that result in the best performance. Saves the best model and its selected variables, to be used for cross validation in Phase C/
## Phase 3.

############################################
###STEP 0: Loading libraries and datasets###
############################################

# install.packages("VGAM")
library(tidyverse) # allows simpler code for dataset manipulation
library(dplyr) # provides tools for more efficient manipulation of datasets
library(VGAM) # allows use of multinomial regression
load("01d-ValidTrainSets.RData")
load("02a-Utils.Rdata")

var_names = names(train)

# Removing any variables that cause multicollinearity and so prevent multinomial logit from working well:
indices_to_remove = which(var_names == "sp_defense") # remove sp_defense - correlated w/ many other base stat variables

train = train[-indices_to_remove]
holdout = holdout[-indices_to_remove]

print(table(holdout$base_egg_steps))


##############################################################
###STEP 1: Fitting multinomial logit model to training data###
##############################################################

multin_logit_model= vglm(base_egg_steps~., multinomial(), data=train)
print(summary(multin_logit_model))


#######################################################################
###STEP 2.1: Getting multinomial logit holdout set class predictions###
#######################################################################

head(holdout)
outpred_multin_logit=predict(multin_logit_model,type="response",newdata=holdout)
round(head(outpred_multin_logit),3)

maxprobMultinLogit=apply(outpred_multin_logit,1,max)
print(summary(maxprobMultinLogit))

sum(maxprobMultinLogit<0.5)
sum(maxprobMultinLogit<0.8)

# Getting category with modal probability for each case:
CatModalProb(outpred_multin_logit)

# How often do these cases match the true values in holdout set?:
print(table(holdout$base_egg_steps, CatModalProb(outpred_multin_logit)))


#####################################################################
###STEP 2.2: Encoding factor levels of response variable to labels###
#####################################################################

# ordered levels of base_egg_steps will be encoded to labels to preserve functionality
# of grepl in Utils.R :
# S : Short , for level "<=3840"
# M : Moderate, for level "5120"
# L : Long, for level "6400"
# E: Extreme, for level ">=7680"

encod_holdo = EncodeToChar(holdout$base_egg_steps, c("S","M","L","E"))
encod_holdo = factor(encod_holdo, levels=c("S","M","L","E"), ordered=TRUE)


###############################################################################
###STEP 2.3: Getting the prediction intervals and their performance measures###
###############################################################################

# To get 50% and 80% prediction intervals:
pred_int_multin_logit=OrdinalPredInterval(outpred_multin_logit,labels=c("S","M","L","E"),
                                       level1=0.5, level2=0.8)

table50 = table(encod_holdo, pred_int_multin_logit$pred1) # w/ 50% pred interval
table80 = table(encod_holdo, pred_int_multin_logit$pred2) # w/ 80% pred interval
print(table50) 
print(table80) 

# Calculating losses for prediction intervals:
intervalLoss50 = PredIntervalLoss(pred_int_multin_logit$pred1,
                                  true_labels=encod_holdo)
intervalLoss80 = PredIntervalLoss(pred_int_multin_logit$pred2,
                                  true_labels=encod_holdo)

# Getting average coverage rate and average length of prediction intervals across all classes:
performance50 = CoverageAcrossClasses(table50) # for 50% pred interval
performance80 = CoverageAcrossClasses(table80) # for 80% pred interval


###########################################################################################
###STEP 3.1: Subsetting train and holdout set variables, needed to run Forward Selection###
###########################################################################################

train_no_respon = subset(train, select = -c(base_egg_steps))
holdo_no_respon = subset(holdout, select = -c(base_egg_steps))
response_train = factor((EncodeToChar(train$base_egg_steps, c("S","M","L","E"))),
                       levels=c("S","M","L","E"),
                       ordered=TRUE)
response_holdo = factor((EncodeToChar(holdout$base_egg_steps, c("S","M","L","E"))),
                        levels=c("S","M","L","E"),
                        ordered=TRUE)


##########################################################
###STEP 3.2: Variable selection using Forward Selection###
##########################################################

## INCLUDE 6 MULTIN_SELECTS !! :

# select variables using 50% prediction interval:
multin_select_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                             always_include = c("base_total","capture_rate"))
# select variables using 80% prediction interval:
multin_select_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                             always_include = c("base_total","capture_rate", "weight_kg"))


#####################################################################################################################################################################
###STEP 4.1: Building encapsulating function to run Multinomial Logistic Regression on any set of selected variables and get pred intervals and class predictions###
#####################################################################################################################################################################

#' @description
#' (Encapsulate running Multinomial Logit on any set of selected variables) - fits new multinomial model, gets holdout class predictions,
#' prediction intervals, and performance measures (interval losses, average lengths, coverage rate)
#' @param formula a formula for vglm e.g. ' base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type '
#' @param use_pred50 true if model is to be based on the 50% prediction intervals, 
#'                    false if model is to be based on the 80% prediction intervals
#' @param train data train set
#' @param holdout data holdout set
#'
#' @return list of fitted model, holdout class predictions, prediction intervals and performance measures.
#'
RunMultinWithSelectedVars = function(formula, use_pred50=TRUE, train, holdout){
   multin_model = vglm(formula, multinomial(), data=train)
   outpred=predict(multin_model,type="response",newdata=holdout)
   pred_int=OrdinalPredInterval(outpred,labels=c("S","M","L","E"), level1=0.5, level2=0.8) # to get both 50 and 80% intervals
   if (use_pred50) {
      pred_int = pred_int$pred1  # if based on 50% pred interval
   } else {
      pred_int = pred_int$pred2  # if based on 80% pred interval
   }
   int_loss = PredIntervalLoss(pred_int, true_labels=encod_holdo) # at our specified % of pred interval
   table_multin = table(encod_holdo, pred_int) # Making table to compare prediction intervals with true holdout categories
   performance = CoverageAcrossClasses(table_multin) # Getting average length and coverage rate from the table
   
   return(list(multin_model = multin_model, outpred = outpred, pred_int=pred_int, int_loss=int_loss,
               table_multin=table_multin, performance=performance))
}




###############################################################################################
###STEP 4.2: Running Multinomial Logit with the above selected variables, and above function###
###############################################################################################

# using our selected variables from Forward Selection based on 50% pred interval above ( multin_select_var50 ) :
select50multin_logit_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, TRUE, train, holdout) # fit model based on 50% pred interval
select50multin_logit_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, FALSE, train, holdout) # fit model based on 80% pred interval

# using our variables from Forward Selection based on 80% pred interval above ( multin_select_var80 ) :
select80multin_logit_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type
                                           +is_bug_type+experience_growth, TRUE, train, holdout)  # fit model based on 50% pred interval
select80multin_logit_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type
                                                 +is_bug_type+experience_growth, FALSE, train, holdout)  # fit model based on 80% pred interval

# using all selected variables (UNION) from Forward Selection based on both 50 and 80% pred intervals above:
select80v50multin_logit_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type+weight_kg+is_normal_type
                                              +is_bug_type+experience_growth, TRUE, train, holdout) # fit model based on 50% pred interval
select80v50multin_logit_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type+weight_kg+is_normal_type
                                                    +is_bug_type+experience_growth, FALSE, train, holdout) # fit model based on 80% pred interval

# using overlapping variables (INTERSECT) from Forward Selection based on both 50 and 80% pred intervals above:
select80n50multin_logit_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate, TRUE, train, holdout) # fit model based on 50% pred interval
select80n50multin_logit_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate, FALSE, train, holdout) # fit model based on 80% pred interval



########################################################################################
###STEP 4.3: Getting new holdout set class predictions using newly selected variables###
########################################################################################

# DELETE THIS WHOLE STEP???

#################################################################################################################
###STEP 4.4: Getting new prediction intervals and performance measures using new holdout set class predictions###
#################################################################################################################

# DELETE THIS WHOLE STEP ????? 

# To get 50% and 80% prediction intervals:
predint50vars_multin=OrdinalPredInterval(outpred50vars_multin,labels=c("S","M","L","E"),
                                        level1=0.5, level2=0.8)
predint80vars_multin=OrdinalPredInterval(outpred80vars_multin,labels=c("S","M","L","E"),
                                         level1=0.5, level2=0.8)
predint80v50vars_multin=OrdinalPredInterval(outpred80v50vars_multin,labels=c("S","M","L","E"),
                                         level1=0.5, level2=0.8)
predint80n50vars_multin=OrdinalPredInterval(outpred80n50vars_multin,labels=c("S","M","L","E"),
                                         level1=0.5, level2=0.8)

# code pasted from earlier, just for reference:
# encod_holdo = EncodeToChar(holdout$base_egg_steps, c("S","M","L","E"))
# encod_holdo = factor(encod_holdo, levels=c("S","M","L","E"), ordered=TRUE)

# Calculating losses for 50 and 80% prediction intervals:
# using just variables from 50% pred intervals earlier
int_loss50_vars50 = PredIntervalLoss(predint50vars_multin$pred1,
                                  true_labels=encod_holdo)
int_loss80_vars50 = PredIntervalLoss(predint50vars_multin$pred2,
                                    true_labels=encod_holdo)
# using just variables from 80% pred intervals earlier
int_loss50_vars80 = PredIntervalLoss(predint80vars_multin$pred1,
                                    true_labels=encod_holdo)
int_loss80_vars80 = PredIntervalLoss(predint80vars_multin$pred2,
                                    true_labels=encod_holdo)
# using all variables (UNION) from both 50 and 80% pred intervals earlier
int_loss50_vars80v50 = PredIntervalLoss(predint80v50vars_multin$pred1,
                                    true_labels=encod_holdo)
int_loss80_vars80v50 = PredIntervalLoss(predint80v50vars_multin$pred2,
                                       true_labels=encod_holdo)
# using overlapping variables (INTERSECT) from both 50 and 80% pred intervals earlier
int_loss50_vars80n50 = PredIntervalLoss(predint80n50vars_multin$pred1,
                                       true_labels=encod_holdo)
int_loss80_vars80n50 = PredIntervalLoss(predint80n50vars_multin$pred2,
                                       true_labels=encod_holdo)

# SUMMARY: int_loss50_vars80 and int_loss50_vars80v50 are the lowest pred interval losses! 

# Making tables to compare prediction intervals with true holdout categories:
table50_vars50 = table(encod_holdo, predint50vars_multin$pred1) # w/ 50% pred interval
table80_vars50 = table(encod_holdo, predint50vars_multin$pred2) # w/ 80% pred interval
print(table50_vars50) 
print(table80_vars50)
table50_vars80 = table(encod_holdo, predint80vars_multin$pred1) # w/ 50% pred interval
table80_vars80 = table(encod_holdo, predint80vars_multin$pred2) # w/ 80% pred interval
#print(table50_vars80)
#print(table80_vars80)
table50_vars80v50 = table(encod_holdo, predint80v50vars_multin$pred1) # w/ 50% pred interval
table80_vars80v50 = table(encod_holdo, predint80v50vars_multin$pred2) # w/ 80% pred interval
table50_vars80n50 = table(encod_holdo, predint80n50vars_multin$pred1)
table80_vars80n50 = table(encod_holdo, predint80n50vars_multin$pred2)


# Getting average lengths and coverage rates from the tables:
perf50_vars50 = CoverageAcrossClasses(table50_vars50) # for 50% pred interval
perf80_vars50 = CoverageAcrossClasses(table80_vars50) # for 80% pred interval
perf50_vars80 = CoverageAcrossClasses(table50_vars80)
perf80_vars80 = CoverageAcrossClasses(table80_vars80)
perf50_vars80v50 = CoverageAcrossClasses(table50_vars80v50)
perf80_vars80v50 = CoverageAcrossClasses(table80_vars80v50)
perf50_vars80n50 = CoverageAcrossClasses(table50_vars80n50)
perf80_vars80n50 = CoverageAcrossClasses(table80_vars80n50)


#########################################################
###STEP 5: Concluding our best multinomial logit model###
#########################################################

# Displaying the above models' losses altogether:
avg_losses = as.matrix(cbind(intervalLoss50, intervalLoss80, select50multin_logit_50int$int_loss, select50multin_logit_80int$int_loss,
                             select80multin_logit_50int$int_loss, select80multin_logit_80int$int_loss, select80v50multin_logit_50int$int_loss,
                             select80v50multin_logit_80int$int_loss, select80n50multin_logit_50int$int_loss, select80n50multin_logit_80int$int_loss))
print(avg_losses)

# SUMMARY: select80multin_logit_50int$int_loss and select80v50multin_logit_50int$int_loss are the lowest pred interval losses! 

# Displaying our earlier models' average scores altogether:
average_performances = as.matrix(cbind(performance50, performance80, select50multin_logit_50int$performance, select50multin_logit_80int$performance,
                                       select80multin_logit_50int$performance, select80multin_logit_80int$performance, select80v50multin_logit_50int$performance,
                                       select80v50multin_logit_80int$performance, select80n50multin_logit_50int$performance, select80n50multin_logit_80int$performance))
print(average_performances)

# CONCLUSIONS:
# Looks like using variables from the select80v50multin_logit_50int model (i.e. "base_total", "capture_rate", "is_dragon_type", "is_ground_type", weight_kg",
# "is_normal_type", "is_bug_type", "experience_growth") gives the lowest interval loss (0.2556) among 50% pred intervals of all our models so far,
# tied with the interval loss of select80MultinLogit_50int. It also gives the single lowest interval loss (0.2949 from select80v50multin_logit_80int) among all 80%
# pred intervals. Thus our best chosen variables here seem to be "base_total", "capture_rate", "is_dragon_type", "is_ground_type", weight_kg",
# "is_normal_type", "is_bug_type", and "experience_growth".

final_selected_vars = c("base_total", "capture_rate", "is_dragon_type", "is_ground_type", 
                      "weight_kg","is_normal_type", "is_bug_type", "experience_growth")

# Looking at this best model's tables that compare its pred intervals with the true holdout categories:
print(select80v50multin_logit_50int$table_multin) # 50% pred interval
print(select80v50multin_logit_80int$table_multin) # 80% pred interval


#######################################################################################################################
###STEP 6: Encapsulate fitting our chosen best multinomial model, to be used for cross validation in Phase C/Phase 3###
#######################################################################################################################

MultinFitter = function(data){
   return(vglm(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type+weight_kg+is_normal_type
               +is_bug_type+experience_growth, multinomial(), data=data))
}

MultinPredictor = function(data, model){
   return(predict(model, type="response", newdata=data))
}


#####################################################################################################################################
###STEP 7: Saving all relevant objects and models, including our best model, its features, and its losses and average performances###
#####################################################################################################################################

save(file="02c-MultinLogitModel.RData", multin_logit_train=multin_logit_model, outpred_multin_logit, pred_int_multin_logit,
     RunMultinWithSelectedVars,
     final_selected_vars,
     select80v50multin_logit_50int, select80v50multin_logit_80int,
     avg_losses,
     average_performances, MultinFitter, MultinPredictor)
