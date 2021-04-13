## CODE FILE 01d: Given cleaned train and validation Pokemon datasets in .csv format, extracts them, further cleans them, and converts
## some features to ordered factors. Saves a new RData object with these two modified datasets.

############################################
###STEP 0: Loading libraries and datasets###
############################################
library(tidyverse) # allows simpler code for dataset manipulation
pokemonValid = read_csv("pokemon_cleaned_validationset.csv")
pokemonTrain = read_csv("pokemon_cleaned_trainingset.csv")


###################################################################################################
###Step 1.1: Removing variables which we do not expect to be helpful, or are unique identifiers ###
###################################################################################################

var_names = names(pokemonTrain)

indices_to_remove = c(1:19) #remove against_type variables and abilities

indices_to_remove = c(indices_to_remove, which(var_names == "japanese_name")) #remove Japanese name (unique identifier)
indices_to_remove = c(indices_to_remove, which(var_names == "classfication")) #remove classification (unique identifier)
indices_to_remove = c(indices_to_remove, which(var_names == "name")) #remove name (unique identifier)
indices_to_remove = c(indices_to_remove, which(var_names == "pokedex_number")) #remove pokedex number (unique identifier)

indices_to_remove = c(indices_to_remove, which(var_names == "generation")) # remove generation (based on prior knowledge, 
                                                                           # we do not expect the time when a Pokemon was created 
                                                                           # to relate to its egg step values)

# renaming train and validation sets with simpler names:
train = pokemonTrain[-indices_to_remove]
holdout = pokemonValid[-indices_to_remove]


#######################################################################
###Step 1.2: Converting some features to factors with ordered levels###
#######################################################################
# The following features in this step were reverted back to either type 'character' or 'integer' upon exporting to .csv files
# at the end of running Code File 01b, so here we revert them back to factor type:

train$base_egg_steps = factor(train$base_egg_steps, levels=c("<=3840","5120","6400",">=7680"), ordered=TRUE)
holdout$base_egg_steps = factor(holdout$base_egg_steps, levels=c("<=3840","5120","6400",">=7680"),
                                ordered=TRUE)

train$percentage_male = factor(train$percentage_male, levels=c("<50","50",">50","agender"), ordered=TRUE)
holdout$percentage_male = factor(holdout$percentage_male, levels=c("<50","50",">50","agender"),ordered=TRUE)

train$experience_growth = factor(train$experience_growth, levels=c('<1M', '1M-1.25M', '>=1.25M'), ordered=TRUE)
holdout$experience_growth = factor(holdout$experience_growth, levels=c('<1M', '1M-1.25M', '>=1.25M'), ordered=TRUE)

train$base_happiness = factor(train$base_happiness, levels=c('<70', '>=70'), ordered=TRUE)
holdout$base_happiness = factor(holdout$base_happiness, levels=c('<70', '>=70'), ordered=TRUE)


###########################################################################
###Step 2: Saving modified train and holdout/validation Pokemon datasets###
###########################################################################

save(file="RDataFiles/ValidTrainSets.RData", train, holdout)
