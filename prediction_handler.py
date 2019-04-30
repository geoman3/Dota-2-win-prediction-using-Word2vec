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

        #Now we want to define a network which will output the supplied embedding for each hero
        #input layer
        emb_input = tf.keras.Input(shape=(1,))
        #output layer
        emb_output = self.compiled_network.layers[2](emb_input)
        #combined network
        embed_network = tf.keras.Model(inputs=emb_input, outputs=emb_output)

    def load_odota_data(self, data):
        dir_teams = data['dire_team']
        rad_teams = data['radiant_team']
        labels = data['radiant_win']

        dir_teams = [team.split(",") for team in dir_teams]
        rad_teams = [team.split(",") for team in rad_teams]
        for team in range(len(dir_teams)):
            dir_teams[team] = [int(hero) for hero in dir_teams[team]
        for team in range(len(rad_teams)):
            rad_teams[team] = [int(hero) for hero in rad_teams[team]
        for i in range(len(dir_teams)):
            embed_sum = np.zeros(shape=(2*self.embedding_handler.embedding_dim))
            rad_sum = np.zeros(shape=self.embedding_handler.embedding_dim)
            [rad_sum=rad_sum+self.embed_network.predict()]
            embed_sum[:self.embedding_handler.embedding_dim] 

    def train_model(self,num_in_batch, num_epochs):
        pass

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass

    