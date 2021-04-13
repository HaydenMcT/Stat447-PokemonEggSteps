## CODE FILE 01e: Using the processed R data, creates some univariate/bivariate summaries of the training data

############################################
###STEP 0: Loading libraries and datasets###
############################################
library(GGally) #necessary for bivariate summary
load("RDataFiles/ValidTrainSets.RData")

#######################################################################
###Step 1: Analyzing univariate, bivariate summary statistics       ###
#######################################################################

summary(train)

# look at bivariate summary statistics for the non-type variables 
# (there are too many type variables to visualize)
#this line takes ~5 minutes to run:
bivariate_summary = ggpairs(train[,1:14], title="correlogram with ggpairs()") 
print(bivariate_summary)
