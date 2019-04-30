import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensoreflow.keras import layers

class PredictionNN:
    
    def __init__(self, embedding_handler):
        self.embedding_handler = embedding_handler
        #input layer
        inputs = tf.keras.Input(shape=((2*embedding_handler.embedding_dim),))
        #output layer
        output = layers.Dense(1, activation='sigmoid')(inputs)
        #combined network
        predictor_network = tf.keras.Model(inputs=inputs,outputs=output)
        predictor_network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.predictor_network = predictor_network

    def load_odota_data(self, data):
        dir_teams = data['dire_team']
        rad_teams = data['radiant_team']
        labels = data['radiant_win']
        dir_teams = [team.split(",") for team in dir_teams]
        rad_teams = [team.split(",") for team in rad_teams]

        #here convert each list of teams to the internal nn integer key
        for team in range(len(rad_teams)):
            rad_teams[team] = [int(hero) for hero in rad_teams[team]
        for i, team in enumerate(rad_teams):
            for j, hero in enumerate(team):
                rad_teams[i][j] = self.embedding_handler.api_to_nn[hero]

        for team in range(len(dir_teams)):
            dir_teams[team] = [int(hero) for hero in dir_teams[team]
        for i, team in enumerate(dir_teams):
            for j, hero in enumerate(team):
                dir_teams[i][j] = self.embedding_handler.api_to_nn[hero]

        #now we add the embeddings for each team of heros into a vector for each game that looks like:
        #[[radiant team embedding sum],[dire team embedding sum]]
        embed_sum = np.zeros(shape=(2*self.embedding_handler.embedding_dim,len(dir_teams)))
        for i in range(len(dir_teams)):

            rad_sum = np.zeros(shape=self.embedding_handler.embedding_dim)
            for hero in rad_teams[i]]:
                rad_sum=rad_sum+self.embedding_handler.get_hero_embedding(hero)
            embed_sum[i,:self.embedding_handler.embedding_dim] = rad_sum

            dir_sum = np.zeros(shape=self.embedding_handler.embedding_dim)
            for hero in dir_teams[i]:
                dir_sum=rad_sum+self.embedding_handler.get_hero_embedding(hero)
            embed_sum[i,self.embedding_handler.embedding_dim:] = dir_sum
        #done!
        self.processed_data = embed_sum
        self.labels = labels
        print("Data successfully loaded, pass self.train_model() to train the prediction model.")

    def train_model(self,num_in_batch, num_epochs):
        self.history = self.predictor_network.fit(
            self.processed_data,
            self.labels,
            batch_size=num_in_batch,
            epochs=num_epochs,
            validation_split=0.1
        )

    def save_model(self, file_name):
        assert type(file_name) is str, "file_name must be a string"
        assert file_name[-3:] == ".h5", "file_name must end in: \".h5\""
        self.compiled_network.save_weights(
            "./models/" + file_name, save_format='h5')
        print("The model has been saved under: " + "./models/" + file_name)

    def load_model(self, file_name):
        assert type(file_name) is str, "file_name must be a string"
        self.compiled_network.load_weights("./models/" + file_name)
        print("Model successfully loaded.")