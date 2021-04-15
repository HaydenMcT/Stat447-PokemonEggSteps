## CODE FILE 2d: Given cleaned train and holdout Pokemon datasets from Code File 01d, fits Ordinal Logistic Regression to the data, selects
## variables that result in the best performance. Saves the best model and its selected variables, to be used for cross validation in Phase C/
## Phase 3.

############################################
###STEP 0.1: Loading libraries and datasets###
############################################

library(MASS) # allows usage of polr function to perform ordinal logistic regression
library(tidyverse) # allows simpler code for dataset manipulation
load("RDataFiles/ValidTrainSets.RData")
load("RDataFiles/Utils.Rdata")

#######################################################################################
###STEP 0.2: Encoding factor levels of response variable to single character labels ###
###          (needed for creation and assessment of prediction intervals)           ###
#######################################################################################

# ordered levels of base_egg_steps will be encoded to labels to preserve functionality
# of grepl in Utils.R :
# S : Short , for level "<=3840"
# M : Moderate, for level "5120"
# L : Long, for level "6400"
# E: Extreme, for level ">=7680"

ord_encod_holdo = EncodeToChar(holdout$base_egg_steps, c("S","M","L","E"))
ord_encod_holdo = factor(ord_encod_holdo, levels=c("S","M","L","E"), ordered=TRUE)

###########################################################################################
###STEP 0.3: Subsetting train and holdout set variables, needed to run Forward Selection###
###########################################################################################

train_no_respon = subset(train, select = -c(base_egg_steps))
holdo_no_respon = subset(holdout, select = -c(base_egg_steps))
response_train = factor((EncodeToChar(train$base_egg_steps, c("S","M","L","E"))),
                        levels=c("S","M","L","E"),
                        ordered=TRUE)
response_holdo = factor((EncodeToChar(holdout$base_egg_steps, c("S","M","L","E"))),
                        levels=c("S","M","L","E"),
                        ordered=TRUE)

#########################################################################################################################################
###STEP 1: Building encapsulating function to run polr on any set of selected variables, getting pred intervals and class predictions ###
#########################################################################################################################################

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

##########################################################
###STEP 2.1: Variable selection using Forward Selection###
##########################################################

# select variables using 50% prediction interval:
ordin_select_var50 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                   required_improvement = 0.00, use_pred50 = TRUE, model= "polr")
# select variables using 80% prediction interval:
ordin_select_var80 = ForwardSelect(train_no_respon, holdo_no_respon, response_train, response_holdo,
                                   required_improvement = 0.00, use_pred50 = FALSE, model= "polr")

###########################################################################################
###STEP 2.2: Running Ordinal Logit with the above selected variables, and above function###
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

#########################################################
###STEP 3: Concluding on our best ordinal logit model ###
#########################################################

# Displaying the above models' losses altogether:
ord_avg_losses = matrix(c(select50_polr_model$interval_losses[[1]], select50_polr_model$interval_losses[[2]],
                             select80_polr_model$interval_losses[[1]], select80_polr_model$interval_losses[[2]], 
                             select80v50_polr_model$interval_losses[[1]], select80v50_polr_model$interval_losses[[2]],
                             select80n50_polr_model$interval_losses[[1]], select80n50_polr_model$interval_losses[[2]]), nrow=2, byrow=FALSE,
                           dimnames = list(c("50% Prediction Interval Loss", "80% Prediction Interval Loss"), c("select50", "select80",
                                                                                                                "union of select50, select80", 
                                                                                                                "intersection of select50, select80"))
                           )
print(ord_avg_losses)

# SUMMARY: The union has the lowest pred interval losses for 50%, and is fairly close to the best for 80%, 
# but select80 does best for 80% and almost the best for 50%

# Displaying our earlier models' average scores altogether:
ord_avg_cvg_len = matrix(c(select50_polr_model$interval_cvg_len[[1]], select50_polr_model$interval_cvg_len[[2]],
                          select80_polr_model$interval_cvg_len[[1]], select80_polr_model$interval_cvg_len[[2]], 
                          select80v50_polr_model$interval_cvg_len[[1]], select80v50_polr_model$interval_cvg_len[[2]],
                          select80n50_polr_model$interval_cvg_len[[1]], select80n50_polr_model$interval_cvg_len[[2]]), nrow=4, byrow=FALSE,
                        dimnames = list(c("50% Prediction Interval Length", "50% Prediction Interval Coverage", "80% Prediction Interval Length", "80% Prediction Interval Coverage"), 
                                        c("select50", "select80", "union of select50, select80", "intersection of select50, select80"))
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
###STEP 4: Encapsulate fitting our chosen best ordinal logistic regression model, to be used for cross validation in Phase C/Phase 3###
#######################################################################################################################################

PolrFitter = function(data){
        return(polr(base_egg_steps~is_rock_type+is_dragon_type+capture_rate+is_ghost_type, data=data))
}
PolrPredictor = function(data, model) {
        return(predict(model, type="prob", newdata=data))
}


#####################################################################################################################################
###STEP 5: Saving all relevant objects and models, including our best model, its features, and its losses and average performances###
#####################################################################################################################################

save(file="RDataFiles/OrdLogitModel.RData", best_polr_model_and_metrics = select80_polr_model, final_selected_ord_vars, PolrFitter, PolrPredictor)
