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


####################################################################################################################################################################
###STEP 3.0: Building encapsulating function to run Multinomial Logistic Regression on any set of selected variables and get pred intervals and class predictions###
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

###########################################################################################
###STEP 3.1: Encoding train and holdout set variables, needed to run Forward Selection###
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
###STEP 3.2a: Variable selection using Forward Selection###
##########################################################

# Here we try forward select from scratch, or with a few variables guaranteed to be included in the model
# Using prior knowledge we suspect that base_total, capture_rate, and weight_kg will be important
# we try try models where we force the inclusion of the top 1, 2, or 3 variables we expect to be in the model
# then compare based on holdout set performance to pick which set of variables to select
# we do this once using 50% prediction intervals, and once using 80% prediction intervals

# select variables using 50% prediction interval:

select_include_novars_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                             required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                                             always_include = c()) # not always including any vars
select_include_1vars_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                           required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                                           always_include = c("base_total"))
select_include_2vars_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                             always_include = c("base_total","capture_rate")) # always include base_total and capture_rate
select_include_3vars_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                           required_improvement = 0.0, use_pred50 = TRUE, model= "multinomial",
                                           always_include = c("base_total","capture_rate", "weight_kg"))

# The above models with 0, 1, or 2 variables forced are identical, so we need only compare the first and third model based on holdout set loss
# variables from select_include_novars_var50 :
select50multin_novars = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_ground_type, use_pred50 = TRUE, train, holdout) # fit model based on 50% pred interval

# variables from select_include_3vars_var50 :
select50multin_3vars = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+speed+is_rock_type+is_dragon_type+base_happiness+is_ice_type,
                                                       use_pred50 = TRUE, train, holdout) # fit model based on 50% pred interval

# Make table of results
matrix(c(select50multin_novars$int_loss, select50multin_3vars$int_loss),
       nrow=1, byrow=TRUE,
       dimnames = list(c("Prediction Interval Loss"), c("No forced inclusion", "Include base_total, capture_rate, weight_kg")))
#From the table above, we do not force the inclusion of any variables.

# select variables using 80% prediction interval:
select_include_novars_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                             required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                                             always_include = c()) # not always including any vars
select_include_1vars_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                           required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                                           always_include = c("base_total"))
select_include_2vars_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                           required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                                           always_include = c("base_total","capture_rate")) # always include base_total and capture_rate
select_include_3vars_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.0, use_pred50 = FALSE, model= "multinomial",
                             always_include = c("base_total","capture_rate", "weight_kg"))


# Using our selected variables from Forward Selection based on 80% pred interval above:
# variables from select_include_novars_var80 :
select80multin_novars = RunMultinWithSelectedVars(base_egg_steps~weight_kg+is_dragon_type+is_fire_type+is_ghost_type, use_pred50 = FALSE, train, holdout)  # fit model based on 80% pred interval

# variables from select_include_2vars_var80 :
select80multin_1vars = RunMultinWithSelectedVars(base_egg_steps~base_total+is_water_type+is_normal_type + is_poison_type + is_dragon_type, FALSE, train, holdout)  # fit model based on 80% pred interval

# variables from select_include_2vars_var80 :
select80multin_2vars = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+is_rock_type, FALSE, train, holdout)  # fit model based on 80% pred interval

# variables from select_include_3vars_var80 :
select80multin_3vars = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type+is_bug_type+experience_growth,
                                                       FALSE, train, holdout)  # fit model based on 80% pred interval

# Make table of results
matrix(c(select80multin_novars$int_loss, select80multin_1vars$int_loss, select80multin_2vars$int_loss, select80multin_3vars$int_loss),
       nrow=1, byrow=TRUE,
       dimnames = list(c("Prediction Interval Loss"), c("No forced inclusion", "Include base_total",
                                                        "Include base_total, capture_rate", "Include base_total, capture_rate, weight_kg")))
#From this table, it is best to include all 3 variables according to the 80% prediction interval loss

######################################################################################################
###STEP 3.2b: Fitting Multinomial Logit using the above selected variables and comparing performance###
######################################################################################################

# Lowest loss for selecting via 50% pred intervals is from select50multin_novars , and lowest loss for selecting via 80% pred intervals is from select80multin_3vars
# Thus we shall also try fitting, at 50% and 80% intervals, a few more new models using the Union and the Intersect of these two models' variable sets.

# Using variables from the Union of those used by select80multin_3vars_50int and select50multin_3vars_80int :
union_performance_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_dragon_type+is_ground_type
                                                          +is_normal_type+is_bug_type+experience_growth,
                                                       TRUE, train, holdout) # fit model based on 50% pred interval
union_performance_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_dragon_type+is_ground_type
                                                    +is_normal_type+is_bug_type+experience_growth,
                                                          FALSE, train, holdout) # fit model based on 80% pred interval

# Using overlapping variables from the Intersect of those used by select80multin_3vars_50int and select50multin_3vars_80int :
intersect_performance_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate,
                                                          TRUE, train, holdout) # fit model based on 50% pred interval
intersect_performance_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate,
                                                        FALSE, train, holdout) # fit model based on 80% pred interval

selected_via_50_performance_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate + is_dragon_type + is_ground_type,
                                                                                            TRUE, train, holdout) # fit model based on 50% pred interval
selected_via_50_performance_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate + is_dragon_type + is_ground_type,
                                                        FALSE, train, holdout) # fit model based on 80% pred interval

selected_via_80_performance_50int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type+is_bug_type+experience_growth,
                                                              TRUE, train, holdout) # fit model based on 50% pred interval
selected_via_80_performance_80int = RunMultinWithSelectedVars(base_egg_steps~base_total+capture_rate+weight_kg+is_normal_type+is_bug_type+experience_growth,
                                                              FALSE, train, holdout) # fit model based on 80% pred interval


#########################################################
###STEP 5: Concluding our best multinomial logit model###
#########################################################

# Comparing all models' interval losses again at 50% intervals, this time including our new models:
losses = matrix(c(selected_via_50_performance_50int$int_loss, selected_via_50_performance_80int$int_loss,
                        selected_via_80_performance_50int$int_loss, selected_via_80_performance_80int$int_loss,
                        union_performance_50int$int_loss, union_performance_80int$int_loss, 
                        intersect_performance_50int$int_loss, intersect_performance_80int$int_loss),
                      nrow=2, byrow=FALSE,
                      dimnames = list(c("50% Prediction Interval Loss", "80% Prediction Interval Loss"), c("select50", "select80",
                                             "union", "intersection")))
print(losses)
# RESULT: The union model performs the best for 50% and 80% prediction intervals


# CONCLUSION: Our optimal set of selected variables would be the union of the best features using the 50% prediction interval loss
# and the best features using the 80% prediction interval loss

final_selected_vars = c("base_total", "capture_rate", "weight_kg", "is_ground_type", "is_dragon_type", "", "is_normal_type",
                        "is_bug_type", "experience_growth")

# Looking at this best model's tables that compare its pred intervals with the true holdout categories:
print(union_performance_50int$table_multin) # 50% pred interval
print(union_performance_80int$table_multin) # 80% pred interval


#######################################################################################################################
###STEP 6: Encapsulate fitting our chosen best multinomial model, to be used for cross validation in Phase C/Phase 3###
#######################################################################################################################

MultinFitter = function(data){
   return(vglm(base_egg_steps~base_total+capture_rate+weight_kg+is_dragon_type+is_ground_type
               +is_normal_type+is_bug_type+experience_growth, multinomial(), data=data))
}

MultinPredictor = function(data, model){
   return(predict(model, type="response", newdata=data))
}


##############################################################################################################################
###STEP 7: Saving all relevant objects and models, including our best model, its features, and its losses, and a function  ###
###        for fitting it on new data                                                                                      ###
##############################################################################################################################

save(file="RDataFiles/MultinLogitModel.RData", union_performance_50int, union_performance_80int,
     multin_final_selected_vars = final_selected_vars,
     multin_losses = losses,
     MultinFitter, MultinPredictor)

