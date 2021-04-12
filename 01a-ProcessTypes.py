## Given the default Pokemon dataset (which represents type as two columns, each with 18 different strings for each type), 
## makes a new dataset which represents the types(s) of each pokemon with 18 different binary columns
##
##

import pandas as pd

ALL_TYPES = ['fire', 'water', 'grass', 'electric', 'ground', 'bug', 'normal', 'ice', 'flying', 'poison', 'fighting', 'rock', 'steel', 'fairy', 'dark', 'ghost', 'dragon', 'psychic']

original_df = pd.read_csv("Pokemon_no_missing_with_can_hatch.csv")

num_observations = len(original_df)

new_df = original_df
new_df = new_df.drop(columns = ['type1', 'type2'])

for pokemon_type in ALL_TYPES:
    column_name = 'is_' + pokemon_type + '_type'
    column_values = [False] * num_observations # initialize the column to be all 0 values

    for i in range(num_observations):
        # if the i'th observation matches type "pokemon_type" then set the appropriate values to 1 according to the type of each observation
        if original_df.at[i, 'type1'] == pokemon_type or original_df.at[i, 'type2'] == pokemon_type: 
            column_values[i] = True    
        #note: (some type 2 values are nan, but that is not a problem, since 'nan' == pokemon_type will just always be false)

    #add column to dataset
    new_df[column_name] = column_values

#add another column to indicate which pokemon have no second type
column_name = 'pure_type'
column_values = [False] * num_observations
for i in range(num_observations):
    if pd.isna(original_df.at[i, 'type2']):
        column_values[i] = True

new_df[column_name] = column_values


new_df.to_csv('pokemon_boolean_types_and_manual_preprocessing.csv')


