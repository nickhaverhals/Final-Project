#
# This is my API file to run the predictions created from the random forest
# model I made in the first part of the project.

# Load packages for data preparation, visualization, and modeling
library(lubridate)
library(tidyverse)
library(ggplot2)
library(tidymodels)
library(ranger)
library(plumber)

#Create data tibble
min_played<-read_csv("../2025-26_NBA_Player_Stats_Per_36.csv")

# Create recipe to be used with both models
mp_recipe <- recipe(MPG ~ `3P` + FT + TRB + AST + STL + BLK + TOV + PF+ PTS + Pos, data = mp_train)|>
  step_dummy(Pos)

#Find out how many cores
cores <- parallel::detectCores()

# Set up model
rf_mod <- 
  rand_forest(mtry = 4, min_n = 3, trees = 1000) |> 
  set_engine("ranger", num.threads = cores, importance = "permutation") |> 
  set_mode("regression")

#Define workflow
final_rf_wf <- workflow() |>
  add_model(rf_mod)|>
  add_recipe(mp_recipe)

#set seed for reproducibility
set.seed(6.28)

# Fit the selected model using the entire dataset
best_overall_fit <- fit(final_rf_wf, data = min_played)

# Define defaults for prediction endpoint
mn_3P  <- mean(min_played$`3P`)
mn_FT  <- mean(min_played$FT)
mn_TRB <- mean(min_played$TRB)
mn_AST <- mean(min_played$AST)
mn_STL <- mean(min_played$STL)
mn_BLK <- mean(min_played$BLK)
mn_TOV <- mean(min_played$TOV)
mn_PF  <- mean(min_played$PF)
mn_PTS  <- mean(min_played$PTS)

#Establish position categories
pos_levels <- unique(min_played$Pos)

# Convert Pos to a factor
min_played$Pos <- factor(min_played$Pos,
                         levels = pos_levels)


#* @apiTitle Haverhals Final Project
#* @apiDescription This is the API site for Nick's Final Project.


#* Return predicted minutes played/game based on user defined parameters.
#* @get /pred
function(`3P` = mn_3P, FT = mn_FT, TRB = mn_TRB, AST = mn_AST, STL = mn_STL, BLK = mn_BLK, TOV = mn_TOV, PF = mn_PF, PTS = mn_PTS, Pos = "SG") {
  query <- tibble(
    `3P` = as.numeric(`3P`),
    FT = as.numeric(FT),
    TRB = as.numeric(TRB),
    AST = as.numeric(AST),
    STL = as.numeric(STL),
    BLK = as.numeric(BLK),
    TOV = as.numeric(TOV),
    PF = as.numeric(PF),
    PTS = as.numeric(PTS),
    Pos = factor(Pos, levels = pos_levels)
  )
  
  pred <- predict(best_overall_fit, new_data = query)
  list(pred_mpg = pred$.pred[[1]])
}

#Send a message
#* @get /info
function(){
  list(
    name = "Nick Haverhals",
    URL = "https://nickhaverhals.github.io/Final-Project/")
}


# Prediction URL for Jayden McDaniels (observed: 34.05 MPG, predicted: 25.5467 MPG)
# http://127.0.0.1:12145/pred?3P=1.8&FT=2.2&TRB=6.7&AST=4&STL=1.3&BLK=0.9&TOV=2.2&PF=3.6&PTS=20.3&Pos=PF
#
# Prediction URL for Ayo Dosunmu (observed: 31.79 MPG, predicted: 24.1035 MPG)
# http://127.0.0.1:12145/pred?3P=2.6&FT=2.0&TRB=4.8&AST=4&STL=1.4&BLK=0.4&TOV=2.7&PF=3.2&PTS=20.4&Pos=SG
#
# Prediction URL for Joan Beringer (observed: 16.84 MPG, predicted: 16.7442)
#http://127.0.0.1:12145/pred?3P=0&FT=2.8&TRB=12.9&AST=0.9&STL=1.1&BLK=3&TOV=1.2&PF=3.3&PTS=10&Pos=C
#