# Dota 2 win prediction using Word2vec

## Some background
Typically [Word2vec by Mikolov et. al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (as the name would imply) is used in natural language processing to encode the semantic differences between words into an n'th dimensional vector space based on their context within a body of text e.g. Dog and cat would be considered similar because of their similar proximity to other words like "Vet" and "Pet". In fact, it was found that similar relationships between words had similar displacements from their counterparts such that semantic concepts about words were captured by individual dimensions of the embedding vector space. However there have been a number ([1](https://github.com/evanthebouncy/dota_hero_semantic_embedding), [2](http://federicov.github.io/word-embeddings-and-dota2.html)) of attempts to apply the Word2vec algorithm to heroes in the game Dota 2 by treating the teams in which the heroes were drafted as the context, through this method a number of effective embeddings were obtained with characteristics like whether the hero was ranged or a carry or a mid captured by individual dimensions.

## Our Goal
The purpose of this project is to use the word2vec technique in a similar manner as before to obtain an embedding for the Dota 2 heroes, and to then use this embedding as features in a model to predict the outcome of a game of Dota based purely off of the draft.

## File Explanations

### embedding_handler.py

This file contains the handler class for our neural network which we do most of the embedding work on e.g. loading data, training the model, saving it etc. the methods on this class give us a concise way of defining and optimising training loops as well as visualising the results

### opendota_scrape.py

This is a simple script that scrapes the open dota API for summary data on dota matches and saves it into a .csv.

### opendota_get_herocsv.py

This is a simple script that grabs the basic information for each hero from the opendota API and again saves it to a csv.

### data_exploration.ipynb

I used this notebook to do some basic analysis on the game data to make sure it wasn't biased in some way that I might not be aware of.

### prediction_handler.py

This file contains the handler class for our game prediction model, where the methods again give a concise way of defining training loops and visualising the results. (Not yet written)

### training_scipt.py

A simple script to train and save a wide scan of models with different embedding dimensions and with different numbers of epochs

### ./models/

This folder contains our saved models

### dota_matches.csv

The raw data scraped from the open dota API

### heroes.csv

The summary data for each hero