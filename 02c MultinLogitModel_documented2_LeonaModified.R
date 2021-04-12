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
load("RDataFiles/ValidTrainSets.RData") # data
load("RDataFiles/Utils.Rdata") # functions

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

# select variables using 50% prediction interval:
select_include_novars_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                             required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                                             always_include = c()) # not always including any vars
select_include_2vars_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                             always_include = c("base_total","capture_rate")) # always include base_total and capture_rate
select_include_3vars_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                           required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                                           always_include = c("base_total","capture_rate", "weight_kg")) # always include base_total, capture_rate and weight_kg

# select variables using 80% prediction interval:
select_include_novars_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                             required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                                             always_include = c()) # not always including any vars
select_include_2vars_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                           required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                                           always_include = c("base_total","capture_rate")) # always include base_total and capture_rate
select_include_3vars_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                             always_include = c("base_total","capture_rate", "weight_kg")) # always include base_total, capture_rate and weight_kg

all_sets_of_select_vars = list(select_include_novars_var50, select_include_2vars_var50, select_include_3vars_var50,
                               select_include_novars_var80, select_include_2vars_var80, select_include_3vars_var80)


####################################################################################################################################################################
###STEP 4.1: Building encapsulating function to run Multinomial Logistic Regression on any set of selected variables and get pred intervals and class predictions###
####################################################################################################################################################################

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


######################################################################################################
###STEP 4.2: Fitting Multinomial Logit using the above selected variables and comparing performance###
######################################################################################################

# # making list of formulas using all our sets of selected variables obtained from STEP 3.2 :
#formulas_for_all_var_sets = list(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, # select_include_novars_50
#                                  base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, # select_include_2vars_var50
#                                  base_egg_steps~base_total+capture_rate+weight_kg+speed+is_rock_type+is_dragon_type+base_happiness
#                                  +is_ice_type,                                                        #select_include_3vars_var50
#                                  base_egg_steps~weight_kg+is_dragon_type+is_fire_type+is_ghost_type, #select_include_novars_var80
#                                  base_egg_steps~base_total+capture_rate+is_rock_type, # select_include_2vars_var80
#                                  base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type+is_bug_type+experience_growth) # select_include_3vars_var80
# 
# # create empty list to add fitted models to later:
# all_models = list()
# 
# # fitting models and getting performance measures using our sets of selected variables, at 50% and 80% pred intervals:
# for (i in 1:length(formulas_for_all_var_sets)){
#    all_models = append(all_models, RunMultinWithSelectedVars(formulas_for_all_var_sets[[i]], TRUE, train, holdout)) # fit models based on 50% pred intervals
#    all_models = append(all_models, RunMultinWithSelectedVars(formulas_for_all_var_sets[[i]], FALSE, train, holdout)) # fit models based on 80% pred intervals
# }


# Using our selected variables from Forward Selection based on 50% pred interval above:
# variables from select_include_novars_var50 :
select50multin_novars_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, TRUE, train, holdout) # fit model based on 50% pred interval
select50multin_novars_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, FALSE, train, holdout) # fit model based on 80% pred interval

# variables from select_include_2vars_var50 :
select50multin_2vars_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, TRUE, train, holdout) # fit model based on 50% pred interval
select50multin_2vars_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, FALSE, train, holdout) # fit model based on 80% pred interval

# variables from select_include_3vars_var50 :
select50multin_3vars_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+speed+is_rock_type+is_dragon_type+base_happiness+is_ice_type,
                                                       TRUE, train, holdout) # fit model based on 50% pred interval
select50multin_3vars_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+speed+is_rock_type+is_dragon_type+base_happiness+is_ice_type,
                                                       TRUE, train, holdout) # fit model based on 80% pred interval


# Using our selected variables from Forward Selection based on 80% pred interval above:
# variables from select_include_novars_var80 :
select80multin_novars_50int = RunMultinWithSelectedVars(base_egg_steps~weight_kg+is_dragon_type+is_fire_type+is_ghost_type, TRUE, train, holdout)  # fit model based on 50% pred interval
select80multin_novars_80int = RunMultinWithSelectedVars(base_egg_steps~weight_kg+is_dragon_type+is_fire_type+is_ghost_type, FALSE, train, holdout)  # fit model based on 80% pred interval

# variables from select_include_2vars_var80 :
select80multin_2vars_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_rock_type, TRUE, train, holdout)  # fit model based on 50% pred interval
select80multin_2vars_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_rock_type, FALSE, train, holdout)  # fit model based on 80% pred interval

# variables from select_include_3vars_var80 :
select80multin_3vars_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type+is_bug_type+experience_growth,
                                                       TRUE, train, holdout)  # fit model based on 50% pred interval
select80multin_3vars_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type+is_bug_type+experience_growth,
                                                       FALSE, train, holdout)  # fit model based on 80% pred interval


# Comparing all models' interval losses at 50% pred intervals:
matrix(c(select50multin_novars_50int$int_loss, select50multin_2vars_50int$int_loss, select50multin_3vars_50int$int_loss,
                select80multin_novars_50int$int_loss, select80multin_2vars_50int$int_loss, select80multin_3vars_50int$int_loss),
       nrow=1, byrow=TRUE,
       dimnames = list(c(), c("select50multin_novars", "select50multin_2vars", "select50multin_3vars",
                              "select80multin_novars", "select80multin_2vars", "select80multin_3vars")))

# Comparing all models' interval losses at 80% pred intervals:
matrix(c(select50multin_novars_80int$int_loss, select50multin_2vars_80int$int_loss, select50multin_3vars_80int$int_loss,
         select80multin_novars_80int$int_loss, select80multin_2vars_80int$int_loss, select80multin_3vars_80int$int_loss),
       nrow=1, byrow=TRUE,
       dimnames = list(c(), c("select50multin_novars", "select50multin_2vars", "select50multin_3vars",
                              "select80multin_novars", "select80multin_2vars", "select80multin_3vars")))

# Lowest loss at 50% pred intervals is from select80multin_3vars_50int , and lowest loss at 80% pred intervals is from select50multin_3vars_80int !
# Thus we shall also try fitting, at 50% and 80% intervals, a few more new models using the Union and the Intersect of these two models' variable sets.

# Using variables from the Union of those used by select80multin_3vars_50int and select50multin_3vars_80int :
union_80v50_3vars_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+speed+is_rock_type+is_dragon_type+base_happiness+is_ice_type
                                                          +is_normal_type+is_bug_type+experience_growth,
                                                       TRUE, train, holdout) # fit model based on 50% pred interval
union_80v50_3vars_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+speed+is_rock_type+is_dragon_type+base_happiness+is_ice_type
                                                          +is_normal_type+is_bug_type+experience_growth,
                                                          FALSE, train, holdout) # fit model based on 80% pred interval

# Using overlapping variables from the Intersect of those used by select80multin_3vars_50int and select50multin_3vars_80int :
intersect_80n50_3vars_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg,
                                                          TRUE, train, holdout) # fit model based on 50% pred interval
intersect_80n50_3vars_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg,
                                                        FALSE, train, holdout) # fit model based on 80% pred interval


#########################################################
###STEP 5: Concluding our best multinomial logit model###
#########################################################

# Comparing all models' interval losses again at 50% intervals, this time including our new models:
losses_50int = matrix(c(select50multin_novars_50int$int_loss, select50multin_2vars_50int$int_loss, select50multin_3vars_50int$int_loss,
                        select80multin_novars_50int$int_loss, select80multin_2vars_50int$int_loss, select80multin_3vars_50int$int_loss,
                        union_80v50_3vars_50int$int_loss, intersect_80n50_3vars_50int$int_loss),
                      nrow=1, byrow=TRUE,
                      dimnames = list(c(), c("select50multin_novars", "select50multin_2vars", "select50multin_3vars",
                                             "select80multin_novars", "select80multin_2vars", "select80multin_3vars",
                                             "union_80v50_3vars", "intersect_80n50_3vars")))
# RESULT: Across 50% pred intervals, union_80v50_3vars_50int gives our new lowest interval loss (0.216) !

# Comparing all models' interval losses again at 80% intervals, this time including our new models:
losses_80int = matrix(c(select50multin_novars_80int$int_loss, select50multin_2vars_80int$int_loss, select50multin_3vars_80int$int_loss,
                        select80multin_novars_80int$int_loss, select80multin_2vars_80int$int_loss, select80multin_3vars_80int$int_loss,
                        union_80v50_3vars_80int$int_loss, intersect_80n50_3vars_80int$int_loss),
                      nrow=1, byrow=TRUE,
                      dimnames = list(c(), c("select50multin_novars", "select50multin_2vars", "select50multin_3vars",
                                             "select80multin_novars", "select80multin_2vars", "select80multin_3vars",
                                             "union_80v50_3vars", "intersect_80n50_3vars")))
# RESULT: And across 80% pred intervals, union_80v50_3vars_80int gives our new lowest interval loss (0.270) !

# Displaying all our models' average scores altogether:
avg_performances_50int = as.matrix(cbind(performance50, performance80, select50multin_novars_50int$performance, select50multin_2vars_50int$performance, select50multin_3vars_50int$performance,
                                       select80multin_novars_50int$performance, select80multin_2vars_50int$performance, select80multin_3vars_50int$performance,
                                       union_80v50_3vars_50int$performance, intersect_80n50_3vars_50int$performance)) # across 50% pred intervals
print(avg_performances_50int)

avg_performances_80int = as.matrix(cbind(performance50, performance80, select50multin_novars_80int$performance, select50multin_2vars_80int$performance, select50multin_3vars_80int$performance,
                                         select80multin_novars_80int$performance, select80multin_2vars_80int$performance, select80multin_3vars_80int$performance,
                                         union_80v50_3vars_80int$performance, intersect_80n50_3vars_80int$performance)) # across 80% pred intervals
print(avg_performances_80int)


# CONCLUSION: Our optimal set of selected variables would be the union of select_include_3vars_var50 and select_include_3vars_var80,
# i.e. "base_total", "capture_rate", "weight_kg", "speed", "is_rock_type", "is_dragon_type", "base_happiness", "is_normal_type",
#      "is_bug_type", "experience_growth", "is_ice_type"

final_selected_vars = c("base_total", "capture_rate", "weight_kg", "speed", "is_rock_type", "is_dragon_type", "base_happiness", "is_normal_type",
                        "is_bug_type", "experience_growth", "is_ice_type")

# Looking at this best model's tables that compare its pred intervals with the true holdout categories:
print(union_80v50_3vars_50int$table_multin) # 50% pred interval
print(union_80v50_3vars_80int$table_multin) # 80% pred interval



# # Displaying the above models' losses altogether:
# avg_losses = as.matrix(cbind(intervalLoss50, intervalLoss80, select50multin_logit_50int$int_loss, select50multin_logit_80int$int_loss,
#                              select80multin_logit_50int$int_loss, select80multin_logit_80int$int_loss, select80v50multin_logit_50int$int_loss,
#                              select80v50multin_logit_80int$int_loss, select80n50multin_logit_50int$int_loss, select80n50multin_logit_80int$int_loss))
# 
# # SUMMARY: select80multin_logit_50int$int_loss and select80v50multin_logit_50int$int_loss are the lowest pred interval losses! 
# 
# # Displaying our earlier models' average scores altogether:
# average_performances = as.matrix(cbind(performance50, performance80, select50multin_logit_50int$performance, select50multin_logit_80int$performance,
#                                        select80multin_logit_50int$performance, select80multin_logit_80int$performance, select80v50multin_logit_50int$performance,
#                                        select80v50multin_logit_80int$performance, select80n50multin_logit_50int$performance, select80n50multin_logit_80int$performance))
# print(average_performances)
# 
# # CONCLUSIONS:
# # Looks like using variables from the select80v50multin_logit_50int model (i.e. "base_total", "capture_rate", "is_dragon_type", "is_ground_type", weight_kg",
# # "is_normal_type", "is_bug_type", "experience_growth") gives the lowest interval loss (0.2556) among 50% pred intervals of all our models so far,
# # tied with the interval loss of select80MultinLogit_50int. It also gives the single lowest interval loss (0.2949 from select80v50multin_logit_80int) among all 80%
# # pred intervals. Thus our best chosen variables here seem to be "base_total", "capture_rate", "is_dragon_type", "is_ground_type", weight_kg",
# # "is_normal_type", "is_bug_type", and "experience_growth".
# 
# final_selected_vars = c("base_total", "capture_rate", "is_dragon_type", "is_ground_type", 
#                       "weight_kg","is_normal_type", "is_bug_type", "experience_growth")

# Looking at this best model's tables that compare its pred intervals with the true holdout categories:
# print(select80v50multin_logit_50int$table_multin) # 50% pred interval
# print(select80v50multin_logit_80int$table_multin) # 80% pred interval


#######################################################################################################################
###STEP 6: Encapsulate fitting our chosen best multinomial model, to be used for cross validation in Phase C/Phase 3###
#######################################################################################################################

MultinFitter = function(data){
   return(vglm(base_egg_steps~base_total+capture_rate+weight_kg+speed+is_rock_type+is_dragon_type+base_happiness+is_ice_type
               +is_normal_type+is_bug_type+experience_growth, multinomial(), data=data))
}

MultinPredictor = function(data, model){
   return(predict(model, type="response", newdata=data))
}


#####################################################################################################################################
###STEP 7: Saving all relevant objects and models, including our best model, its features, and its losses and average performances###
#####################################################################################################################################

save(file="RDataFiles/MultinLogitModel.RData", multin_logit_train=multin_logit_model, outpred_multin_logit, pred_int_multin_logit,
     RunMultinWithSelectedVars,
     union_80v50_3vars_50int, union_80v50_3vars_80int,
     final_selected_vars,
     losses_50int, losses_80int,
     avg_performances_50int, avg_performances_80int,
     MultinFitter, MultinPredictor)

