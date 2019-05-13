# Dota 2 win prediction using Word2vec

## Some background
Typically [Word2vec by Mikolov et. al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (as the name would imply) is used in natural language processing to encode the semantic differences between words into an n'th dimensional vector space based on their context within a body of text. To put it in concrete terms, word2vec will attempt to learn some n-dimensional vector representation of each word in your vocabulary such that similar words like 'Guitar' and 'Violin' will have vectors close to each other, but dissimilar words like 'Anger' and 'Shoe' will be further apart. It learns these semantic differences by looking at the words that surround a target word, so 'Violin' and 'Guitar' would be similar because they both appear next to the same words a lot, ('Play','Music'... etc.) and because 'Anger' and 'Shoe' do not appear next to the same words frequently the model will treat them as unrelated. However there have been a number ([1](https://github.com/evanthebouncy/dota_hero_semantic_embedding), [2](http://federicov.github.io/word-embeddings-and-dota2.html)) of attempts to apply the Word2vec algorithm to heroes in the game Dota 2 by treating the teams in which the heroes were drafted as the 'context' and individual heroes as the 'words'. With this technique [evanthebouncy](https://github.com/evanthebouncy) and [Federico](http://federicov.github.io/author/federico-vaggi.html) were able obtain useful embeddings of the Dota 2 hero pool, and even managed to capture concepts about the heroes in individual dimensions of the embedding, e.g. Federico found that the 46th dimension of his embedding space strongly correllated with 'support' heroes.

## The Goal
The purpose of this project is to use the Word2vec technique in a similar manner as before to obtain an embedding for the Dota 2 heroes, and to then use this embedding as features in a model to predict the outcome of a game of Dota based purely off of the draft. The idea being, that teams with a better composition of 'support', 'carry', 'damage', 'durable' etc. roles will have better winrates than those that do not.

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

This file contains the handler class for our game prediction model, where the methods again give a concise way of defining training loops and visualising the results. The prediciton class combines the embedding vectors on each team by summing them together and then using each team's hero-vector-sum as the inputs.

### training_scipt.py

A simple script to train and save a wide scan of models with different embedding dimensions and with different numbers of epochs

### ./models/

This folder contains our saved models

### dota_matches.csv

The raw data scraped from the open dota API

### heroes.csv

The summary data for each hero

## Update:
I have been experimenting with different embedding dimensions and have found that around 16 - 18 dimensions tends to give the most coherent results in terms of outputting sensible heroes from the k_closest_heroes method, however I have been having little success with using the embeddings in a win prediction network, achieving accuracies of ~54%-55% which is about as good as guessing radiant every time. I am experimenting with different netowrk architectures, different ways of combining the embeddings, and just downloading more data.

## Update 2:
After extensive testing I was unable to improve the prediction model beyond 55% validation accuracy using the word2vec embeddings, however when I tried training the embeddings directly on the prediction network, performance increased to 58%. While this result is unsatisfying it is informative, it is possible (expected even) that future models out-perform my own, however it is also possible that this indicates that the draft might not have as strong an effect on the outcome of a match as one might think. Many factors affect the outcome of a game of dota that are not accounted for in this model (like player skill, player behaviour score etc.) so this could serve as a good starting point for future models.
