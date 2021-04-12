## CODE FILE 03a: Creates functions needed to run cross validation

load("RDataFiles/Utils.RData")


#############################
### Cross-Validation Code ###
#############################
#' 
#' @param dataset the full dataset to split for k-fold cross-validation
#' 
#' @param num_folds the number of folds to have
#' 
#' @param seed a random seed to allow replication of the same folds across runs
#' 
#' @returns a list of indices to make train and test folds amenable to k-fold cross validation
#'  (we only return the test indices, because the training set is defined as everything other 
#'   than the test set for each fold)
MakeFolds = function(dataset, num_folds = 4, seed=447){
  n = dim(dataset)[1]
  
  test_fold_idcs = list()
  set.seed(seed)
  shuffled_indices = sample(n) #with a single argument, sample(n) just gives a random permutation of the integers from 1 to n
  
  for (fold in 1:num_folds){
    lower_threshold = round( (fold - 1) * n/num_folds) + 1
    upper_threshold = round(fold * n/num_folds)
    test_indices = shuffled_indices[lower_threshold:upper_threshold]
    test_fold_idcs[[fold]] = test_indices
  } 
  
  return(test_fold_idcs)
}

#' @description
#' (consolidates the logic of evaluating each potential forest/binomial model with cross validation)
#' given a model and a function for evaluating that model, evaluates it using cross-validation 
#' (reports average performance alongside performance on each fold) 
#' 
#' @param dataset full dataset (before cross-fold splitting), possibly with some features removed 
#'                (so that we can test different feature selection models).
#'                response variable is assumed to be the last column of the dataset.
#' 
#' @param model_fitter a function whose only required input is a data subset, and which outputs a classification model for 
#'                     that dataset where everything but the last row is a predictor, and the last row is a response.
#'                     glm and randomForest have both been tested, but other functions should also work
#'                     
#' @param predict_fn function whose only required inputs are a data subset and a model of the type made by model_fitter, and which gives
#'                   prediction probabilities of the response variable for each data point in the provided data subset input.
#'                   
#' @param loss_fn   function which takes predicted probabilities, as well as the true labels, 
#'                   for some set of examples and returns a numeric loss for the model (like a prediction interval loss)
#'                
#' @param seed    the random seed to be used to make folds for cross validation. If this value is the same across runs, 
#'                and so is the ordering of examples in the parameter dataset,
#'                then the same examples will be present in the same folds. This allows comparison between different methods, 
#'                even if some variables are eliminated from the dataset in between runs)
#'                
#' @param num_folds   the number of folds to make (default 3)
#'
#' @return the test performance (as measured by loss_fn) of the model (as specified by model_fitter) across all folds
#'         (with folds being splits of the dataset, as randomized by seed) as well as a vector showing the test
#'         performance for each fold
#'
KFoldCrossValidate = function(dataset, ModelFitter, PredictFn, LossFn = GetLoss, seed=447, num_folds=4){
  
  ### create folds
  test_fold_idcs = MakeFolds(dataset, num_folds, seed)
  
  ### get loss for each fold
  loss_by_fold = c()
  for (fold in 1:num_folds){
    test_indices = test_fold_idcs[[fold]]
    test = dataset[test_indices,]
    train = dataset[-test_indices,]
    
    model = ModelFitter(train)
    preds = PredictFn(test, model)
    loss = LossFn(preds, test$base_egg_steps)
    
    loss_by_fold = append(loss_by_fold, loss)
  }
  
  avg_loss = sum(loss_by_fold)/num_folds
  
  return(list(avg_loss, loss_by_fold))
}

######################################
### Alternative Evaluation Metrics ###
######################################

#' @description
#'  Given the true labels of a response variable, and some model's predicted probabilities 
#'  of each example, returns a numeric score function for the model (in this case, accuracy)
#'                   
#' @param prediction_probs the predicted probability of each example for each class. Assumed that classes are ordinal, and 
#'                         columns of predicted probabilities follow this order                 
#'              
#' @param labels a vector containing the true labels for each example. Assumed to be ordinal
#'               
#'              
#' @return AUC for the model that generated prediction_probs
#'
GetAccuracy = function(prediction_probs, labels){
  prediction_idcs = CatModalProb(prediction_probs)
  classes = levels(labels)
  predictions = classes[prediction_idcs]
  score = mean(labels == predictions)
  return(score)
}

###################################################
### Cross Validation for evaluation via a table ###
###################################################

#' @description
#'  Given the true labels of a response variable, and some model's predicted probabilities 
#'  of each example, returns a table evaluating the model (specifically, a table summarizing how far
#'  the prediction intervals made from the provided probabilities are from containing the true class)
#'                   
#' @param prediction_probs the predicted probability of each example for each class. Assumed that classes are ordinal, and 
#'                         columns of predicted probabilities follow this order                 
#'              
#' @param labels a vector containing the true labels for each example. Assumed to be ordinal
#' 
#' @param Use_pred50 a boolean which is true if the table should be based on 50% prediction intervals, or false if we should use 80% prediction intervals
#'               
#'              
#' @return AUC for the model that generated prediction_probs
TableMaker = function(pred_probs, true_labels, use_pred50 = TRUE) {
  pred_int = OrdinalPredInterval(pred_probs,labels=c("S", "M", "L", "E"))
  if (use_pred50) {
    pred_int = pred_int$pred1
  } else {
    pred_int = pred_int$pred2
  }
  return(PredIntMisclassTab(pred_int, true_labels))
}

#' @description
#' given a model and a function for evaluating that model in table form, evaluates it using cross-validation 
#' (reports a table for each fold as well as a table with each cell averaged across folds) 
#' 
#' @param dataset full dataset (before cross-fold splitting), possibly with some features removed 
#'                (so that we can test different feature selection models).
#'                response variable is assumed to be the last column of the dataset.
#' 
#' @param ModelFitter a function whose only required input is a data subset, and which outputs a classification model for 
#'                     that dataset where everything but the last row is a predictor, and the last row is a response.
#'                     glm and randomForest have both been tested, but other functions should also work
#'                     
#' @param PredictFn function whose only required inputs are a data subset and a model of the type made by ModelFitter, and which gives
#'                   prediction probabilities of the response variable for each data point in the provided data subset input.
#'                   
#' @param TableFn   function which takes predicted probabilities, as well as the true labels, 
#'                   for some set of examples and returns a numeric loss for the model (like a prediction interval loss)
#'                
#' @param seed    the random seed to be used to make folds for cross validation. If this value is the same across runs, 
#'                and so is the ordering of examples in the parameter dataset,
#'                then the same examples will be present in the same folds. This allows comparison between different methods, 
#'                even if some variables are eliminated from the dataset in between runs)
#'                
#' @param num_folds   the number of folds to make (default 4)
#'
#' @return the test performance in table form (as measured by TableFn) of the model (as specified by ModelFitter) across all folds
#'         (with folds being splits of the dataset, as randomized by seed) as well as a vector showing the test
#'         performance for each fold
#'
KFoldCrossValidateTable = function(dataset, ModelFitter, PredictFn, TableFn = TableMaker, seed=447, num_folds=4){
  
  ### create folds
  test_fold_idcs = MakeFolds(dataset, num_folds, seed)
  
  ### get loss for each fold
  table_by_fold = list()
  for (fold in 1:num_folds){
    test_indices = test_fold_idcs[[fold]]
    test = dataset[test_indices,]
    train = dataset[-test_indices,]
    
    model = ModelFitter(train)
    preds = PredictFn(test, model)
    tab = TableFn(preds, test$base_egg_steps)
    
    table_by_fold[[fold]] = tab
  }
  
  avg_table = AverageTables(table_by_fold)
  
  return(list(avg_table, table_by_fold))
}

save(file="RDataFiles/CrossValidationTools.RData", MakeFolds, KFoldCrossValidate, GetLoss, GetAccuracy, TableMaker, KFoldCrossValidateTable)
