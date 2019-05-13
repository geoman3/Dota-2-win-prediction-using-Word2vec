import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class PredictionNN:
    
    def __init__(self, embedding_handler):
        self.embedding_handler = embedding_handler
        num_heroes = len(embedding_handler.heroes_df['id'])
        embedding_weights = embedding_handler.compiled_network.layers[2].get_weights()
        #this network seemed to perform the best after 3 epochs with ~730k training examples
        #input layer
        inputs = tf.keras.Input(shape=(10,))
        #embedding layer
        embedding = layers.Embedding(num_heroes,15,input_length=10,trainable=True)(inputs)
        flattener = layers.Flatten()(embedding)
        #hidden layer - not finalised still tweaking parameters
        hidden1 = layers.Dense(150,activation='relu')(flattener)
        dropout1 = layers.Dropout(0.2)(hidden1)
        hidden2 = layers.Dense(150,activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.2)(hidden2)
        hidden3 = layers.Dense(75,activation='relu')(dropout2)
        hidden4 = layers.Dense(40,activation='relu')(hidden3)
        #output layer
        output = layers.Dense(1, activation='sigmoid')(hidden4)
        #combined network
        predictor_network = tf.keras.Model(inputs=inputs,outputs=output)
        predictor_network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        print(predictor_network.summary())
        self.predictor_network = predictor_network

    def load_odota_data(self, data):
        dir_teams = list(data['dire_team'])
        rad_teams = list(data['radiant_team'])
        labels = list(data['radiant_win'])
        dir_teams = [team.split(",") for team in dir_teams]
        rad_teams = [team.split(",") for team in rad_teams]

        #here convert each list of teams to the internal nn integer key
        for team in range(len(rad_teams)):
            rad_teams[team] = [int(hero) for hero in rad_teams[team]]
            
        for i, team in enumerate(rad_teams):
            for j, hero in enumerate(team):
                rad_teams[i][j] = self.embedding_handler.api_to_nn[hero]

        for team in range(len(dir_teams)):
            dir_teams[team] = [int(hero) for hero in dir_teams[team]]
        for i, team in enumerate(dir_teams):
            for j, hero in enumerate(team):
                dir_teams[i][j] = self.embedding_handler.api_to_nn[hero]

        processed_data = np.array(list(map(list.__add__, rad_teams, dir_teams)))
        print(processed_data[:5])
        #done!
        self.processed_data = processed_data
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
        self.predictor_network.save_weights(
            "./models/" + file_name, save_format='h5')
        print("The model has been saved under: " + "./models/" + file_name)

    def load_model(self, file_name):
        assert type(file_name) is str, "file_name must be a string"
        self.predictor_network.load_weights("./models/" + file_name)
        print("Model successfully loaded.")