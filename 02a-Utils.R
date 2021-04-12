## CODE FILE 2a: Creates new Rdata object containing all utility functions needed for Phase B,
## and some for Phase C.


#' @description 
#'  Find category with modal probability
#' @param predMatrix ncases x J matrix; J is number of response categories
#' Each row of predMatrix is a probability mass function.
#' @return vector of length ncases, each entry in {1,...,J}
CatModalProb=function(predMatrix)
{ modevalue=function(pmf) { tem=which(pmf==max(pmf)); return(tem[1]) }
# return first category in case of ties
apply(predMatrix,1,modevalue)
}



#' @description
#' Prediction intervals for a categorical response
#' @param ProbMatrix of dimension nxJ, J = # categories,
#' each row is a probability mass function
#' @param labels vector of length J, with short names for categories
#' @param level1 numeric decimal representing first level of pred interval
#' @param level2 numeric decimal representing second level of pred interval
#'
#' @details
#' level1 and level2 as params allows this to be a more general function for any
#' level(s) of prediction intervals.
#'
#' @return list with two string vectors of length n:
#' pred1 has level1 prediction intervals
#' pred2 has level2 prediction intervals
#'
CategoryPredInterval = function(ProbMatrix,labels, level1, level2)
{ ncases=nrow(ProbMatrix)
pred1=rep(NA,ncases); pred2=rep(NA,ncases)
for(i in 1:ncases)
{ p=ProbMatrix[i,]
ip=order(p)
pOrdered=p[ip] # increasing order
labelsOrdered=labels[rev(ip)] # decreasing order
G=rev(cumsum(c(0,pOrdered))) # cumulative sum from smallest
k1=min(which(G<=(1-level1)))-1 # 1-level1
k2=min(which(G<=(1-level2)))-1 # 1-level2
predlevel1 =labelsOrdered[1:k1]; predlevel2 =labelsOrdered[1:k2]
pred1[i]=paste(predlevel1,collapse="")
pred2[i]=paste(predlevel2,collapse="")
}
list(pred1=pred1, pred2=pred2)
}




#' @description
#' Prediction intervals for an ordinal response
#' @param ProbMatrix of dimension nxJ, J = # categories,
#' each row is a probability mass function
#' @param labels vector of length J, with short names for categories
#' @param level1 numeric decimal representing first level of pred interval
#' @param level2 numeric decimal representing second level of pred interval
#'
#' @details
#' level1 and level2 as params allows this to be a more general function for any
#' level(s) of prediction intervals.
#'
#' @return list with two string vectors of length n:
#' pred1 has level1 prediction intervals
#' pred2 has level2 prediction intervals
#' Predicted categories are NOT ordered by decreasing probability
#'
OrdinalPredInterval = function(ProbMatrix,labels, level1=.50, level2=.80)
{ 
nlabels = ncol(ProbMatrix)
ncases=nrow(ProbMatrix)
pred1=rep(NA,ncases); pred2=rep(NA,ncases)
for(i in 1:ncases)
{ p=ProbMatrix[i,]
# try the best size 1 interval, then size 2 if the best size 1 interval 
# doesn't capture (1-level1)% confidence, and so on
for(interval_size in 1:nlabels){
  candidate = FindMaxContiguousSubset(p, interval_size)
  if (candidate$sum > level1){
    pred_level1 = labels[candidate$indices]
    break
  }
}
#find interval for second pred_level
for(interval_size in 1:nlabels){
  candidate = FindMaxContiguousSubset(p, interval_size)
  if (candidate$sum > level2){
    pred_level2 = labels[candidate$indices]
    break
  }
}
pred1[i]=paste(pred_level1,collapse="")
pred2[i]=paste(pred_level2,collapse="")
}
list(pred1=pred1, pred2=pred2)
}



#' @description
#' Given a vector p and a length len, find the maximal contiguous interval of length len. Assumes maximal interval is >0
#' @param p a vector for which to find the maximal contiguous subset (in some applications, this will be a probability)
#' @param len the length of the contiguous subset to find. Must be <= length(p) and >= 1
#' 
#' @return The value and indices for the maximal contiguous interval of length len
#'
FindMaxContiguousSubset = function(p, len)
{
num_entries = length(p)
#check for errors in input values
if (num_entries < len){
  print("Error! number of entries in provided array p is less than requested length of interval.")
  return(list(sum=0,indices=0))
}
if (len < 1){
  print("Error! requested length of interval must be an integer >= 1.")
  return(list(sum=0,indices=0))
}
#find maximal subset
best_sum = 0
best_interval_indices = c()
for(i in 1:(num_entries - len + 1)){
  current_sum = sum(p[i:(i+len-1)])
  if (current_sum > best_sum){
    best_sum = current_sum
    best_interval_indices = i:(i+len-1)
  }
}
return(list(sum = best_sum, indices = best_interval_indices))
}



#' @description
#' Create contingency table showing the frequencies some given prediction intervals: contain the true class, miss the true class by 1 class,
#' miss the true class by 2 classes, or miss the true class by 3 classes.
#' @param prediction_intervals vector of length n, giving prediction intervals for each data point
#'                            e.g. 50% pred intervals for each point
#' @param true_labels vector of length n, giving true labels for each holdout data point. If having k ordered levels, vector should have been encoded to
#'                    k factored AND ORDERED labels.
#'
#' @return Contingency table showing numbers of correct and 'off' (by how many classes) classifications.
#'
PredIntMisclassTab = function(prediction_intervals,true_labels){
  tab <- matrix(rep(0,16), ncol=4, byrow=TRUE)
  colnames(tab) <- c("good", "missed_by_1", "missed_by_2", "missed_by_3")
  rownames(tab) <- c("S", "M", "L", "E")
  tab<-as.table(tab)
  
  n = length(prediction_intervals)
  level_vector = levels(true_labels)
  klevels = length(levels(true_labels))
  
  for (i in 1:n){
      interval = prediction_intervals[i]
      if (grepl(true_labels[i], interval)){ # i.e. if pred interval contains true value
        tab[true_labels[i],"good"] <- tab[true_labels[i],"good"] + 1
      }
      else if (true_labels[i]==level_vector[1] & ( grepl(level_vector[2],interval) )){
        tab[true_labels[i],"missed_by_1"] <- tab[true_labels[i],"missed_by_1"]+1
      }
      else if (true_labels[i]==level_vector[2] & ( grepl(level_vector[1],interval) | grepl(level_vector[3],interval) )){
        tab[true_labels[i],"missed_by_1"] <- tab[true_labels[i],"missed_by_1"]+1
      }
      else if (true_labels[i]==level_vector[3] & ( grepl(level_vector[2],interval) | grepl(level_vector[4],interval) )){
        tab[true_labels[i],"missed_by_1"] <- tab[true_labels[i],"missed_by_1"]+1
      }
      else if (true_labels[i]==level_vector[4] & ( grepl(level_vector[3],interval) )){
        tab[true_labels[i],"missed_by_1"] <- tab[true_labels[i],"missed_by_1"]+1
      }
      else if (true_labels[i]==level_vector[1] & ( grepl(level_vector[3],interval) )){
        tab[true_labels[i],"missed_by_2"] <- tab[true_labels[i],"missed_by_2"]+1
      }
      else if (true_labels[i]==level_vector[2] & ( grepl(level_vector[4],interval) )){
        tab[true_labels[i],"missed_by_2"] <- tab[true_labels[i],"missed_by_2"]+1
      }
      else if (true_labels[i]==level_vector[3] & ( grepl(level_vector[1],interval) )){
        tab[true_labels[i],"missed_by_2"] <- tab[true_labels[i],"missed_by_2"]+1
      }
      else if (true_labels[i]==level_vector[4] & ( grepl(level_vector[2],interval) )){
        tab[true_labels[i],"missed_by_2"] <- tab[true_labels[i],"missed_by_2"]+1
      }
      else if (true_labels[i]==level_vector[1] & ( grepl(level_vector[4],interval) )){  # missing it completely, nowhere close.
        tab[true_labels[i],"missed_by_3"] <- tab[true_labels[i],"missed_by_3"]+1
      }
      else if (true_labels[i]==level_vector[4] & ( grepl(level_vector[1],interval) )){  # missing it completely, nowhere close.
        tab[true_labels[i],"missed_by_3"] <- tab[true_labels[i],"missed_by_3"]+1
      }
  }
  return(tab)
}



#' @description
#' Encode length-n vector of K ORDERED factor levels, to K ordered single characters as factor labels.
#' @param ordFact vector of length n with K ORDERED factor levels
#' @param newLabels K-length vector of ORDERED single-character labels for each ordered factor
#'
#' @details
#' newLabels as params allows this to be a more general function for any chosen factor labels.
#'
#' @return encod_labels vector of length n, with new labels for each ordered factor
#'
EncodeToChar = function(ordFact, newLabels)
{
ncases = length(ordFact)
klevels = length(levels(ordFact))
encod_labels=rep(NA,ncases)
for(i in 1:ncases){
  for(j in 1:klevels){
    if (ordFact[i]==(levels(ordFact))[j]){
      encod_labels[i]=paste(newLabels[j])
    }
  }
}
return(encod_labels)
}



#' @description
#' Calculate loss for prediction intervals with a categorical response that has 4 categories
#' @param prediction_interval vector of length n, giving prediction intervals for each data point
#' @param true_labels vector of length n, giving true labels for each data point
#' @param costs_correct vector giving costs for having the correct letter in the prediction interval, for intervals containing 1,2, 3, or 4 categories respectively
#'                      (note that costs_correct[1] is always assumed to be 0, because that corresponds to correctly predicting the exact category)
#' @param costs_incorrect vector giving costs for not having the correct letter in the interval, for intervals containing 1,2, 3, or 4 categories respectively
#'                        (note that we will never have an incorrect prediction if we include all 4 categories in the interval, so costs_incorrect[4] is meaningless)
#'
#' @return The loss for the provided prediction interval (a higher value means the prediction interval is worse)
#'
PredIntervalLoss = function(prediction_intervals,true_labels, 
                            costs_correct = c(0, 1/4, 2/4, 3/4) * 1/length(true_labels), 
                            costs_incorrect = c(1, 1, 1, 0) * 1/length(true_labels)){
  n = length(true_labels)
  loss = 0 #initialize loss function at 0
  for (i in 1:n) {
    interval = prediction_intervals[i]
    interval_size = nchar(interval)
    is_correct = grepl(true_labels[i], interval) #check if prediction interval contains true value
    if (is_correct){
      loss = loss + costs_correct[interval_size]
    } else {
      loss = loss + costs_incorrect[interval_size]
    }
  }
  return(loss)
}



#' @description Coverage rate of prediction intervals for a categorical response
#' @param Table table with true class labels as row names, pred intervals as column names
#' @return list with average length, #misses, miss rate, coverage rate by class
Coverage=function(Table)
{ nclass=nrow(Table); npred=ncol(Table); rowFreq=rowSums(Table)
labels=rownames(Table); predLabels=colnames(Table)
cover=rep(0,nclass); avgLen=rep(0,nclass)
for(irow in 1:nclass)
{ for(icol in 1:npred)
{ intervalSize = nchar(predLabels[icol])
isCovered = grepl(labels[irow], predLabels[icol])
frequency = Table[irow,icol]
cover[irow] = cover[irow] + frequency*isCovered
avgLen[irow] = avgLen[irow] + frequency*intervalSize
}
}
miss = rowFreq-cover; avgLen = avgLen/rowFreq
out=list(avgLen=avgLen,miss=miss,missRate=miss/rowFreq,coverRate=cover/rowFreq)
return(out)
}



#' @description Returns True Coverage rate of prediction intervals for a categorical response
#'              not coverage rate per class
#' @param Table table with true class labels as row names, pred intervals as column names
#' @return coverage rate and average length of prediction interval across all classes
CoverageAcrossClasses =function(Table)
{ nclass=nrow(Table); npred=ncol(Table); n = sum(Table)
labels=rownames(Table); predLabels=colnames(Table)
cover=0; avgLen=0
for(irow in 1:nclass)
{ for(icol in 1:npred)
{ intervalSize = nchar(predLabels[icol])
isCovered = grepl(labels[irow], predLabels[icol])
frequency = Table[irow,icol]
cover = cover + frequency*isCovered
avgLen = avgLen + frequency*intervalSize
}
}
avgLen = avgLen/n
out=list(avgLen=avgLen,coverRate=cover/n)
return(out)
}



#' @description
#' (wrapper for predInterval loss) - calculates loss for a prediction interval made from the provided predictions
#' @param predictions matrix of length n x 3, giving predictions for each data point, 
#'                    where a prediction is a probability for each possible letter
#' @param true_labels vector of length n, giving true labels for each data point
#' @param use_pred50 true if loss is to be based on the 50% prediction intervals, 
#'                    false if loss is to be based on the 80% prediction intervals
#' 
#'
#' @return loss based on a prediction interval formed from predictions (higher value means a worse model)
#'
GetLoss = function(predictions, true_labels, use_pred50=TRUE){
  predInt = OrdinalPredInterval(predictions,labels=c("S", "M", "L", "E"))
  if (use_pred50){
    loss = PredIntervalLoss(predInt$pred1, true_labels)
  } else {
    loss = PredIntervalLoss(predInt$pred2, true_labels)
  }
  return(loss)
}



#' @description
#' (consolidates the logic of evaluating each potential tree/multinomial model)
#' given a model, fits that model to the training set and creates predictions such that each category has a predicted probability
#' @param model one of "random_forest", "multinomial", "polr", "ctree", corresponding to the model to be used
#' @param train_set training set, possibly with some features removed (so that we can test different feature selection models)
#' @param train_y response variable for training set
#' @param predict_set set to evaluate predictions on. May be the same as training set, or could match a validation/holdout set
#' 
#' @return predictions of the newly fit model on predict_set, where the prediction for each example are formatted as a probability
#'         of that example matching each class, respectively
#'
GetModelPreds = function(model, train_set, train_y, predict_set){
  if (model == "multinomial"){
    multinom = vglm(train_y ~ ., multinomial(), data=train_set)
    preds = predict(multinom, type="response", newdata=predict_set)
  } else if (model == "random_forest"){
    rforest = randomForest(train_y ~ .,data=train_set, importance=TRUE, proximity=TRUE)
    preds = predict(rforest, type="prob", newdata=predict_set)
  } else if (model == "ctree"){
    ctree = rpart(train_y ~ .,data=train_set)
    preds = predict(ctree, newdata=predict_set)
  } else if (model=='polr'){
    ordLogit= polr(train_y ~., data=train_set)  # Using 3 most important predictors
    preds=predict(ordLogit,type="probs",newdata=predict_set)
  }else {
    print("model not recognized. Should be one of \"multinomial\", \"random_forest\", \"polr\", or \"ctree\". ")
    return(0)
  }
  return(preds)
}



#' @description
#' Performs forward selection: at each step, it adds the one variable
#' which decreases loss the most for this multinomial logistic classifier's prediction interval
#' given the previously selected variables. Stops adding variables when
#' validation/holdout set accuracy stops improving.
#'
#' @param train the dataset on which to fit the data. CANNOT INCLUDE RESPONSE VBL
#'              (all features will be considered in the model)
#' @param validation the dataset on which to test the model
#' @param y a vector representing the response variable for this dataset
#' @param y_tilde a vector representing the response variable for the validation set
#' @param required_improvement a vector representing how much training set improvement is required to warrant considering a new variable
#' @param use_pred50 if true, use a 50% prediction interval. Else use 80%
#' @param model one of "random_forest", "multinomial", "ctree", corresponding to the model to be used for feature selection
#' @param always_include a vector of variable names to always select as part of the model 
#' (it is assumed that always_include is a subset of the variable names in train) 
#' @return the variables selected by the model
ForwardSelect = function(train, validation, y, y_tilde, required_improvement = 0.00, use_pred50 = TRUE, model= "multinomial", always_include = c()){
  var_names=names(train)
  max_score = -Inf
  max_validation_score = -Inf
  variables = always_include
  var_to_add = var_names[1]
  add_new = TRUE
  while(add_new == TRUE){
    add_new = FALSE
    for (variable in var_names[!(var_names %in% variables)]){
      predictions = GetModelPreds(model, subset(train, select = c(variable, variables)), y, train)
      loss = GetLoss(predictions, y, use_pred50)
      model_score = 1 - loss
      if (model_score > max_score + required_improvement){
        add_new = TRUE
        max_score = model_score
        var_to_add = variable
      }
    }
    if (add_new){
      predictions_validation = GetModelPreds(model, subset(train, select = c(var_to_add, variables)), y, validation)
      loss = GetLoss(predictions_validation, y_tilde, use_pred50)
      new_validation_score = 1 - loss
      if (new_validation_score > max_validation_score){
        variables = c(variables, var_to_add)
        max_validation_score = new_validation_score
      } else {
        add_new = FALSE
      }
    }
  }
  return(variables)
}



# needed for Phase C
#' @description
#' Produces an nxn table with averages of values from some given nxn tables.
#' @param tables k-length list of nxn tables.
#'
#' @return table of averaged values from all given tables.
#'
AverageTables = function(tables){
  ktables = length(tables)
  ncols = ncol(tables[[1]])
  tab <- matrix(rep(0,ncols*ncols), ncol=ncols, byrow=TRUE)
  colnames(tab) <- c("good", "missed_by_1", "missed_by_2", "missed_by_3")
  rownames(tab) <- c("S", "M", "L", "E")
  tab<-as.table(tab)
  
  for (k in 1:ktables){
    for(i in 1:ncols){
      for(j in 1:ncols){
        tab[i,j] = tab[i,j] + (1/ktables)*(tables[[k]][i,j])
      }
    }
  }
  return(tab)
}





save(file="RDataFiles/Utils.RData", CoverageAcrossClasses, CatModalProb, CategoryPredInterval, OrdinalPredInterval, FindMaxContiguousSubset, PredIntervalLoss, Coverage,
     EncodeToChar, ForwardSelect, GetModelPreds, GetLoss, PredIntMisclassTab, AverageTables)

