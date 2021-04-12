### CODE FILE 02e: Run a basic, greedy decision tree algorithm on the Pokemon dataset (we'll likely get an interpretable, but low accuracy, model)
###

############################################
###STEP 0: Loading libraries and datasets###
############################################
library(rpart)       # allows use of decision trees
library(rpart.plot)  # allows plotting of decision trees

load("RDataFiles/validTrainSets.RData")
load("RDataFiles/Utils.RData")

############################################
###STEP 1: Hyperparameter tuning         ###
############################################

# 4 key hyperparameters control the performance of decision trees for rpart:
# minbucket (the minimum number of training examples required to be within each leaf 
# - so each leaf must be responsible for classifying at least minbucket training examples)
# minsplit (the minimum number of training examples required to pass through each internal node)
# maxdepth (the maximum depth of the decision tree)
# cp - the complexity parameter. A minimum required increase in "fit" for each split (this seems to correspond to an internal R^2 value for anova splitting)

# these all define stopping criteria for the algorithm, allowing it not to consider certain splts if they break requirements for maximum
# depth, minsplit, minbucket, or cp.

#we try a few values here, and pick the best.

best_loss_50 = 1
best_tree_50 = rpart(base_egg_steps ~., data=train)

best_tree_80 = best_tree_50
best_loss_80 = 1

best_depth = c(0, 0)
best_bucket = c(0, 0)
best_cp = c(0, 0)

#hyperparameter search - takes a few minutes to run
for (cp in c(.5, 0.1, 0.05, 0.01, 0.005, 0.001)){
  for (bucket_size in c(2, 4, 8, 16)){
    for (depth in 3:12){
      ctree = rpart(base_egg_steps ~., data=train, minbucket = bucket_size, maxdepth=depth, cp=cp)
      
      holdout_pred_probs=predict(ctree,newdata=holdout)
      pred_int = OrdinalPredInterval(holdout_pred_probs, labels = c("S","M", "L", "E"))
      loss_50 = PredIntervalLoss(pred_int$pred1, EncodeToChar(holdout$base_egg_steps, c("S","M","L","E")))
      loss_80 = PredIntervalLoss(pred_int$pred2, EncodeToChar(holdout$base_egg_steps, c("S","M","L","E")))
        
      if (loss_50 < best_loss_50){
        best_tree_50 = ctree
        best_loss_50 = loss_50
        best_depth[1] = depth
        best_cp[1] = cp
        best_bucket[1] = bucket_size
      }
      if (loss_80 < best_loss_80){
        best_tree_80 = ctree
        best_loss_80 = loss_80
        best_depth[2] = depth
        best_cp[2] = cp
        best_bucket[2] = bucket_size
      }
    }
  }
}

############################################
###STEP 2: Model Selection               ###
############################################

#define a function to analyze prediction interval loss: 
#' @description
#' calculates 50% and 80% prediction interval loss for a given decision tree model, based on the holdout set
#' @param model a trained decision tree model
#'
#' @return loss based on a prediction interval formed from the model's predictions on the holdout set
#'        (higher value means a worse model)
#'
EvaluateTree = function(model){
  holdout_pred_probs=predict(model,newdata=holdout)
  pred_int = OrdinalPredInterval(holdout_pred_probs, labels = c("S","M", "L", "E"))
  loss_50 = PredIntervalLoss(pred_int$pred1, EncodeToChar(holdout$base_egg_steps, c("S","M","L","E")))
  loss_80 = PredIntervalLoss(pred_int$pred2, EncodeToChar(holdout$base_egg_steps, c("S","M","L","E")))
  return(c(loss_50, loss_80))
}

#best_tree_80 is a bit simpler and a bit less awful for the 6400 class, so we use it
losses_best_50 = EvaluateTree(best_tree_50)
losses_best_80 = EvaluateTree(best_tree_80)

matrix(c(losses_best_50, losses_best_80),
       nrow=2, byrow=FALSE,
       dimnames = list(c("Prediction Interval Loss (50%)", "Prediction Interval Loss (80%)"), c("Best Tree (50%)", "Best Tree (80%)")))

#since best_tree_80 performs much better on the 80% prediction interval and nly a ittle worse at 50%, use it.
ctree = best_tree_80

############################################################################################
###STEP 3: Save/Plot  Model, as well as saving functions to fit it on future data        ###
############################################################################################
rpart.plot(ctree)


#' @description
#' Fits previously selected classification tree model (see cart.R) to the provided dataset
#' 
#' @param dataset Dataset on which to fit the model. response variable is assumed to be 
#'                the last column of the dataset, and every other column is assumed to be
#'                an explanatory variable.
#'                
#' @return a classification tree model object, fit to the provided dataset and capable of making new predictions
#'   
CTreeFitter = function(data, formula = base_egg_steps ~ .) {
  return(rpart(formula,data=data, minbucket = 4, maxdepth=8, cp=0.01))
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
CTreePredictor = function(data, model) {
  return(predict(model, newdata=data))
}

save(file="RDataFiles/DecisionTreeModel.Rdata", ctree, CTreeFitter, CTreePredictor)