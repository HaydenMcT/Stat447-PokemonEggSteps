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

Thank you!