## CODE FILE 2d: Given cleaned train and holdout Pokemon datasets from Code File 01d, fits Ordinal Logistic Regression to the data, selects
## variables that result in the best performance. Saves the best model and its selected variables, to be used for cross validation in Phase C/
## Phase 3.

############################################
###STEP 0: Loading libraries and datasets###
############################################

library(MASS) # allows usage of polr function to perform ordinal logistic regression
library(tidyverse) # allows simpler code for dataset manipulation
load("RDataFiles/ValidTrainSets.RData")
load("RDataFiles/Utils.Rdata")

var_names = names(train)

# Removing any variables that cause multicollinearity and so prevent ordinal logit from working well:
indices_to_remove = which(var_names == "sp_defense") # remove sp_defense; correlated w/ many other base stat variables

train = train[-indices_to_remove]
holdout = holdout[-indices_to_remove]


###########################################################
###STEP 1: Fitting ordinal logit model to training data ###
###########################################################

ord_logit= polr(base_egg_steps~attack+weight_kg+capture_rate, data=train)  # Using 3 most important predictors
print(summary(ord_logit))


###################################################################
###STEP 2.1: Getting ordinal logit holdout set class predictions###
###################################################################

outpred_ord_logit=predict(ord_logit,type="probs",newdata=holdout)
print(round(head(outpred_ord_logit),3))

max_prob_ord_logit=apply(outpred_ord_logit,1,max)
print(summary(max_prob_ord_logit))

sum(max_prob_ord_logit<0.5)
sum(max_prob_ord_logit<0.8)

# Getting category with modal probability for each case:
CatModalProb(outpred_ord_logit)

# How often do these cases match the true values in holdout set?:
print(table(holdout$base_egg_steps, CatModalProb(outpred_ord_logit)))
# category 3 never gets to be the category with modal probability!


#####################################################################
###STEP 2.2: Encoding factor levels of response variable to labels###
#####################################################################

# ordered levels of base_egg_steps will be encoded to labels to preserve functionality
# of grepl in Utils.R :
# S : Short , for level "<=3840"
# M : Moderate, for level "5120"
# L : Long, for level "6400"
# E: Extreme, for level ">=7680"

ord_encod_holdo = EncodeToChar(holdout$base_egg_steps, c("S","M","L","E"))
ord_encod_holdo = factor(ord_encod_holdo, levels=c("S","M","L","E"), ordered=TRUE)


###############################################################################
###STEP 2.3: Getting the prediction intervals and their performance measures###
###############################################################################

# to get 50% and 80% prediction intervals:
pred_int_ord_logit=OrdinalPredInterval(outpred_ord_logit,labels=c("S","M","L","E"))

ord_table50 = table(ord_encod_holdo, pred_int_ord_logit$pred1) # w/ 50% pred interval
ord_table80 = table(ord_encod_holdo, pred_int_ord_logit$pred2) # w/ 80% pred interval
print(ord_table50) 
print(ord_table80) 


# Calculating losses for prediction intervals:
ord_loss50 = PredIntervalLoss(pred_int_ord_logit$pred1,
                                  true_labels=ord_encod_holdo)
ord_loss80 = PredIntervalLoss(pred_int_ord_logit$pred2,
                                  true_labels=ord_encod_holdo)


# Getting average coverage rate and average length of prediction intervals across all classes:
ord_perf50 = CoverageAcrossClasses(ord_table50) # for 50% pred interval
ord_perf80 = CoverageAcrossClasses(ord_table80) # for 80% pred interval


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
ordin_select_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.00, use_pred50 = TRUE, model= "polr")
# select variables using 80% prediction interval:
ordin_select_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                             required_improvement = 0.00, use_pred50 = FALSE, always_include= c("base_total", "capture_rate", "weight_kg"), model= "polr")


#####################################################################################################################################################################
###STEP 4.1: Building encapsulating function to run Ordinal Logistic Regression on any set of selected variables, getting pred intervals and class predictions###
#####################################################################################################################################################################

#' @description
#' (Encapsulate running Ordinal Logit on any set of selected variables) - fits new ordinal regression model, gets holdout class predictions,
#' prediction intervals, and performance measures (interval losses, average lengths, coverage rate) using the standard train/holdout split in this file
#' 
#' @param formula a formula for polr e.g. ' base_egg_steps~base_total+capture_rate+is_dragon_type+is_rock_type+is_water_type+percentage_male '
#'                    
#' @return list of fitted model, holdout class predictions, prediction intervals and performance measures.
#'
RunOrdWithSelectedVars = function(formula){
        ordin_model = polr(formula, data=train)
        outpred=predict(ordin_model,type="probs",newdata=holdout)
        pred_int=OrdinalPredInterval(outpred,labels=c("S","M","L","E"), level1=0.5, level2=0.8) # to get both 50 and 80% intervals
        intervals = list(pred_int$pred1, pred_int$pred2) #format prediction intervals as a list
        
        #find performance measures for each interval:
        interval_losses = list()
        interval_tables = list()
        interval_cvg_len = list()
        for (idx in 1:2) {
                interval = intervals[[idx]]
                interval_losses[[idx]] = PredIntervalLoss(interval, true_labels=ord_encod_holdo) # at our specified % of pred interval
                interval_tables[[idx]] = table(ord_encod_holdo, interval) # Making table to compare prediction intervals with true holdout categories
                interval_cvg_len[[idx]] = CoverageAcrossClasses(interval_tables[[idx]]) # Getting average length and coverage rate from the table
        }
        
        return(list(ordin_model = ordin_model, outpred = outpred, intervals = intervals, interval_losses = interval_losses, 
                    interval_tables = interval_tables, interval_cvg_len = interval_cvg_len))
}

###########################################################################################
###STEP 4.2: Running Ordinal Logit with the above selected variables, and above function###
###########################################################################################

# using our selected variables from Forward Selection based on 50% pred interval above ( ordin_select_var50 ) :
select50_polr_model = RunOrdWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_rock_type)

# using our selected variables from Forward Selection based on 80% pred interval above ( ordin_select_var80 ) :
select80_polr_model = RunOrdWithSelectedVars(base_egg_steps~is_rock_type+is_dragon_type+capture_rate+is_ghost_type)

# using all selected variables (UNION) from Forward Selection based on both 50 and 80% pred intervals above:
select80v50_polr_model = RunOrdWithSelectedVars(base_egg_steps~base_total+capture_rate+is_dragon_type+is_rock_type
                                                        +is_water_type+percentage_male+is_ghost_type)

# using overlapping variables (INTERSECT) from Forward Selection based on both 50 and 80% pred intervals above:
select80n50_polr_model = RunOrdWithSelectedVars(base_egg_steps~is_rock_type+is_dragon_type+capture_rate)

#####################################################
###STEP 5: Concluding our best ordinal logit model###
#####################################################

# Displaying the above models' losses altogether:
ord_avg_losses = matrix(c(ord_loss50, ord_loss80, select50_polr_model$interval_losses[[1]], select50_polr_model$interval_losses[[2]],
                             select80_polr_model$interval_losses[[1]], select80_polr_model$interval_losses[[2]], 
                             select80v50_polr_model$interval_losses[[1]], select80v50_polr_model$interval_losses[[2]],
                             select80n50_polr_model$interval_losses[[1]], select80n50_polr_model$interval_losses[[2]]), nrow=2, byrow=FALSE,
                           dimnames = list(c("50% Prediction Interval Loss", "80% Prediction Interval Loss"), c("all variables","select50", "select80",
                                                                                                                "union of select50, select80", 
                                                                                                                "intersection of select50, select80"))
                           )
print(ord_avg_losses)

# SUMMARY: The union has the lowest pred interval losses for 50%, and is fairly close to the best for 80%, 
# but select80 does best for 80% and almost the best for 50%

# Displaying our earlier models' average scores altogether:
ord_avg_cvg_len = matrix(c(ord_perf50, (ord_perf80), select50_polr_model$interval_cvg_len[[1]], select50_polr_model$interval_cvg_len[[2]],
                          select80_polr_model$interval_cvg_len[[1]], select80_polr_model$interval_cvg_len[[2]], 
                          select80v50_polr_model$interval_cvg_len[[1]], select80v50_polr_model$interval_cvg_len[[2]],
                          select80n50_polr_model$interval_cvg_len[[1]], select80n50_polr_model$interval_cvg_len[[2]]), nrow=4, byrow=FALSE,
                        dimnames = list(c("50% Prediction Interval Length", "50% Prediction Interval Coverage", "80% Prediction Interval Length", "80% Prediction Interval Coverage"), 
                                        c("all variables","select50", "select80", "union of select50, select80", "intersection of select50, select80"))
)
print(ord_avg_cvg_len)

# CONCLUSIONS:
# select80 gives the lowest 80% pred interval loss (0.3314) among all other models' 80% pred intervals, and
# gives a 50% interval loss (0.2725) that's NEAR the lowest 50% interval loss.
# On the other hand, select80v50 gives the lowest 50% pred interval loss (0.2612) among all other models' 50% pred intervals, and
# gives an 80% interval loss (0.3427) that's NEAR the lowest 80% interval loss.
# Between the two models, their coverage rates are quite similar when comparing both their 50% and 80% pred intervals,
# but because select80 uses fewer predictors, we choose select80 here as the best model to use. i.e. use
# variables "is_rock_type", "is_dragon_type", "capture_rate" and "is_ghost_type".

final_selected_ord_vars = c("is_rock_type", "is_dragon_type", "capture_rate","is_ghost_type")

# Looking at this best model's tables that compare its pred intervals with the true holdout categories:
print(select80_polr_model$interval_tables[[1]]) # 50% pred interval
print(select80_polr_model$interval_tables[[2]]) # 80% pred interval


#######################################################################################################################################
###STEP 6: Encapsulate fitting our chosen best ordinal logistic regression model, to be used for cross validation in Phase C/Phase 3###
#######################################################################################################################################

PolrFitter = function(data){
        return(polr(base_egg_steps~is_rock_type+is_dragon_type+capture_rate+is_ghost_type, data=data))
}
PolrPredictor = function(data, model) {
        return(predict(model, type="prob", newdata=data))
}


#####################################################################################################################################
###STEP 7: Saving all relevant objects and models, including our best model, its features, and its losses and average performances###
#####################################################################################################################################

save(file="RDataFiles/OrdLogitModel.RData", select80_polr_model, final_selected_ord_vars, PolrFitter, PolrPredictor)
