## CODE FILE 03b: Runs cross validation on all models for a variety of metrics

###################################################
###STEP 0: Loading files and Processing Dataset ###
###################################################
load("RDataFiles/Utils.RData")
load("RDataFiles/CrossValidationTools.RData")

load("RDataFiles/MultinLogitModel.RData")
load("RDataFiles/OrdLogitModel.RData")
load("RDataFiles/RandomForestModel.RData")
load("RDataFiles/DecisionTreeModel.RData")

load("RDataFiles/ValidTrainSets.RData")
dataset = rbind(train, holdout)
dataset$base_egg_steps = factor((EncodeToChar(dataset$base_egg_steps,
                                              c("S","M","L","E"))),
                                levels=c("S","M","L","E"),
                                ordered=TRUE)


######################################################################
###STEP 1: Evaluating Each Model from Step 2 with Cross Validation ###
######################################################################
library(rpart)
cross_val_result_ctree_50 = KFoldCrossValidate(dataset, CTreeFitter, CTreePredictor, num_folds = 4)
cross_val_result_ctree_80 = KFoldCrossValidate(dataset, CTreeFitter, CTreePredictor, 
                                                  LossFn = function(preds, true_labels){
                                                    return(GetLoss(preds, true_labels, use_pred50 = FALSE))
                                                    })

cross_val_pt_acc_ctree = KFoldCrossValidate(dataset, CTreeFitter, CTreePredictor, GetAccuracy, num_folds = 4)

table_summary_ctree_50 = KFoldCrossValidateTable(dataset, CTreeFitter, CTreePredictor, num_folds = 4)
table_summary_ctree_80 = KFoldCrossValidateTable(dataset, CTreeFitter, CTreePredictor, 
                                               TableFn = function(preds, true_labels){
                                                       return(TableMaker(preds, true_labels, use_pred50 = FALSE))
                                               })

library(VGAM)
cross_val_result_multin_50 = KFoldCrossValidate(dataset, MultinFitter, MultinPredictor, num_folds = 4)
cross_val_result_multin_80 = KFoldCrossValidate(dataset, MultinFitter, MultinPredictor, 
                                                  LossFn = function(preds, true_labels){
                                                    return(GetLoss(preds, true_labels, use_pred50 = FALSE))
                                                  })
cross_val_pt_acc_multin = KFoldCrossValidate(dataset, MultinFitter, MultinPredictor, GetAccuracy, num_folds = 4)

table_summary_multin_50 = KFoldCrossValidateTable(dataset, MultinFitter, MultinPredictor, num_folds = 4)
table_summary_multin_80 = KFoldCrossValidateTable(dataset, MultinFitter, MultinPredictor, 
                                                     TableFn = function(preds, true_labels){
                                                             return(TableMaker(preds, true_labels, use_pred50 = FALSE))
                                                     })



library(MASS)
cross_val_result_polr_50 = KFoldCrossValidate(dataset, PolrFitter, PolrPredictor, num_folds = 4)
cross_val_result_polr_80 = KFoldCrossValidate(dataset, PolrFitter, PolrPredictor, 
                                                   LossFn = function(preds, true_labels){
                                                     return(GetLoss(preds, true_labels, use_pred50 = FALSE))
                                                   })
cross_val_pt_acc_polr = KFoldCrossValidate(dataset, PolrFitter, PolrPredictor, GetAccuracy, num_folds = 4)

table_summary_polr_50 = KFoldCrossValidateTable(dataset, PolrFitter, PolrPredictor, num_folds = 4)
table_summary_polr_80 = KFoldCrossValidateTable(dataset, PolrFitter, PolrPredictor, 
                                                     TableFn = function(preds, true_labels){
                                                             return(TableMaker(preds, true_labels, use_pred50 = FALSE))
                                                     })

library(randomForest)
cross_val_result_RF_50 = KFoldCrossValidate(dataset, RFFitter, RFPredictor, num_folds = 4)
cross_val_result_RF_80 = KFoldCrossValidate(dataset, RFFitter, RFPredictor, 
                                                 LossFn = function(preds, true_labels){
                                                   return(GetLoss(preds, true_labels, use_pred50 = FALSE))
                                                 })

cross_val_pt_acc_RF = KFoldCrossValidate(dataset, RFFitter, RFPredictor, GetAccuracy, num_folds = 4)

table_summary_RF_50 = KFoldCrossValidateTable(dataset, RFFitter, RFPredictor, num_folds = 4)
table_summary_RF_80 = KFoldCrossValidateTable(dataset, RFFitter, RFPredictor, 
                                                     TableFn = function(preds, true_labels){
                                                             return(TableMaker(preds, true_labels, use_pred50 = FALSE))
                                                     })


save(file="RDataFiles/ComputeCV.RData", cross_val_result_ctree_50, cross_val_result_ctree_80, 
     cross_val_result_multin_50, cross_val_result_multin_80, 
     cross_val_result_polr_50, cross_val_result_polr_80, 
     cross_val_pt_acc_ctree, cross_val_pt_acc_multin, cross_val_pt_acc_polr,
     cross_val_pt_acc_RF,
     cross_val_result_RF_50, cross_val_result_RF_80, table_summary_ctree_50, table_summary_ctree_80,
     table_summary_multin_50, table_summary_multin_80, table_summary_polr_50, table_summary_polr_80, table_summary_RF_50, table_summary_RF_80)
