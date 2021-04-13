# Stat 447 - Team 2 Final Submission

Hello, and welcome to the codebase for Team 2's Stat 447 submission.

If you would like to run the codebase from start to finish, each
file is ordered from 01a through to 03b. Our initial processing and
exploration of the dataset is in python (both a .py code file and some Jupyter notebooks), 
but after that we use exclusively R. If you do not wish to run all of these files, 
we have included the intermediate data created by each file, so that the required inputs 
for each file are already present, even if the preceding files are not run

This analysis is based on the dataset found at https://www.kaggle.com/rounakbanik/pokemon
As there are several missing values, and we also needed a column to differentiate from 
Pokemon which can and cannot hatch from eggs, we manually entered the missing values
and information about which pokemon can and cannot hatch from eggs by using the data
on https://www.serebii.net/pokedex-sm/
Because of this preprocessing, the first csv file is NOT the raw dataset from Kaggle,
but rather our manually updated version of that dataset, without missing values and with
an extra column indicating which Pokemon can hatch: pokemon_no_missing_with_can_hatch.csv

No files (that are not in ArchivedCode) take >5 minutes to run, although some take several minutes. 

The files, in the order in which they should be run, are:

01a-ProcessTypes.py : given the manually preprocessed starting dataset pokemon_no_missing_with_can_hatch.csv 
(which represents type as two columns, each with 18 different strings for each type), 
this file makes a new dataset which represents the types(s) of each pokemon with 19 different binary columns 
(a column for each type, and a 19th column to indicate if the Pokemon has only one type).

01b-ExploratoryAnalysis-Part1.ipynb : Given the manually processed Pokemon dataset, performs 
data cleaning, feature conversion, and data splitting into train and validation sets, 
and makes 2 new .csv files of each of these two new sets.

01c-ExploratoryAnalysis-Part2.ipynb : Given the cleaned Pokemon training dataset from 
Exploratory Analysis Pt. 1, plot explanatory variables against the response 
(base_egg_steps) to visualize relationships between them. Perform a variety of exploratory analyses.

01d-ValidTrainSets.R : Given cleaned train and validation Pokemon datasets in .csv format, extracts them, 
further cleans them, and converts some features to ordered factors. Saves a new RData object with these 
two modified datasets.

02a-Utils.R : Creates new Rdata object containing utility functions needed for the rest of the project, like
computing prediction intervals and assigning prediction intervals a loss value.

02b-RandomForestModel.R : Given cleaned train and holdout Pokemon datasets from Code File 01d, fits random forest models to the data and selects
variables that result in the best performance. Saves the best model and its selected variables, along with functions to be used 
when fitting the model on new data and making predictions. These functions are later used for cross validation.

02c-MultinLogitModel.R : Given cleaned train and holdout Pokemon datasets from Code File 01d, fits multinomial logistic regression models to the data and selects
variables that result in the best performance. Saves the best model and its selected variables, along with functions to be used 
when fitting the model on new data and making predictions. These functions are later used for cross validation.

02d-OrdLogitModel.R : Given cleaned train and holdout Pokemon datasets from Code File 01d, fits proportional odds logistic regression to the data and selects
variables that result in the best performance. Saves the best model and its selected variables, along with functions to be used 
when fitting the model on new data and making predictions. These functions are later used for cross validation.

02e-DecisionTreeModel.R : Given cleaned train and holdout Pokemon datasets from Code File 01d, uses a greedy decision tree algorithm to fit a model to the data, 
selecting hyperparameters using the holdout set so that the model performs reasonably well. We also create functions to be used 
when fitting the model (with the selected hyperparameters) on new data and making predictions. These functions are later used for cross validation.

03a-CrossValidationTools.R : Creates functions needed to run cross validation

03b-ComputeCV.R : Runs cross validation on all models for a variety of metrics

In ArchivedCode, you can find code which was used as a part of our process when working
on the project, but which is not necessary for the final report. This includes 02f-KNNmodel.R ,
which fits a KNN model (which we found to haave unacceptably low performance) and 01e-SummarizeTrain.R, 
which reports several univariate and bivariate statistics for the training set but which takes
a long time to run, and is not an essential part of our process.

Thank you!