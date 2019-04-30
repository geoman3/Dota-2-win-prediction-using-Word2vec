import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils


class EmbeddingNN:

    def __init__(self, heroes_df, embedding_dim):
        assert type(
            embedding_dim) is int, "embedding dimension must be an integer"
        self.embedding_dim = embedding_dim
        self.heroes_df = heroes_df
        #api_to_nn[hero representation] = internal embedding id
        #nn_to_api[internal embedding id] = hero representation
        #The api keys supplied by open dota do not align with the index of the hero,
        #e.g. Mars' api key is 129 but there are only 117 heroes so these dictionaries will map each key to
        #an integer between 1 and 117 so the embedding will not have extraneous nodes.
        self.api_to_nn = {}
        self.nn_to_api = {}
        self.nn_to_name = {}
        self.name_to_nn = {}
        
        #here we assemble the necessary dictionaries to convert between the different hero representations
        for i in range(len(heroes_df['id'])):
            self.api_to_nn[heroes_df['id'][i]] = i+1
            self.nn_to_api[i+1] = heroes_df['id'][i]
            self.nn_to_name[i+1] = heroes_df['localized_name'].iloc[i]
            self.name_to_nn[heroes_df['localized_name'].iloc[i]] = i+1

        #Assembling the untrained model
        #Here we assemble the embedding layer to take in the hero and its teammate as inputs
        embedding_layer = layers.Embedding(len(heroes_df['id']),embedding_dim,input_length=1)

        #Main hero input
        target_hero_input = tf.keras.Input(shape=(1,))
        target_hero_embedding = embedding_layer(target_hero_input)
        target_hero_embedding = layers.Reshape((embedding_dim,1))(target_hero_embedding)

        #The main hero's teammate's hero input
        teammate_hero_input = tf.keras.Input(shape=(1,))
        teammate_hero_embedding = embedding_layer(teammate_hero_input)
        teammate_hero_embedding = layers.Reshape((embedding_dim,1))(teammate_hero_embedding)

        #merging the outputs of the embedding layer outputs
        merged_embedding_dotproduct = keras.layers.dot(
            inputs=[target_hero_embedding, teammate_hero_embedding],
            axes=1,
            normalize=False
        )
        merged_embedding_dotproduct = layers.Reshape(
            (1,))(merged_embedding_dotproduct)

        output_node = layers.Dense(1, activation="sigmoid")(
            merged_embedding_dotproduct)

        model = keras.Model(
            [target_hero_input, teammate_hero_input], output_node)
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        self.compiled_network = model

        # Here we define a secondary model which we can pass a pair of heroes into to get a measure
        # of how similar their embeddings are (bigger number = more similar).
        cos_similarity_layer = keras.layers.dot(
            inputs=[target_hero_embedding, teammate_hero_embedding],
            axes=0,
            normalize=True
        )
        self.cos_sim_model = keras.Model(
            [target_hero_input, teammate_hero_input], cos_similarity_layer)

        print(model.summary())
        print("Model succesfully assembled, pass self.load_data(data) to prepare your data for training")

    def load_odota_data(self, data):
        data = list(data)

        # The summary data we pulled from the open dota api gave the heroes as a single string with the
        # api keys separated by commas, so we need to convert this string of api keys to a list of integers
        # that correspond to the index of the hero
        data = [team.split(",") for team in data]
        for team in range(len(data)):
            data[team] = [int(hero) for hero in data[team]]
        for i, team in enumerate(data):
            for j, hero in enumerate(team):
                data[i][j] = self.api_to_nn[hero]
        couples_list = []
        labels_list = []
        
        # the skipgrams keras function generates a sequence of pairs from the supplied data and labels it 1 or 0
        # where a pair labeled 1 the teammate was on the target hero's team, and 0 if the pair did not appear in
        # in the target hero's team
        for team in data:
            couples, labels = keras.preprocessing.sequence.skipgrams(
                sequence=team,
                vocabulary_size=len(self.api_to_nn),
                window_size=len(team),
                negative_samples=1
            )
            couples_list += couples
            labels_list += labels
        target_hero, teammate_hero = zip(*couples_list)
        self.target_hero = np.array(target_hero, dtype="int32")
        self.teammate_hero = np.array(teammate_hero, dtype="int32")
        self.labels = labels_list
        print("Data successfully loaded into model object, pass self.train_model() to train the model")

    def train_model(self, num_in_batch, num_epochs):
        self.history = self.compiled_network.fit(
            (self.target_hero, self.teammate_hero),
            self.labels,
            batch_size=num_in_batch,
            epochs=num_epochs,
            validation_split=0.1
        )
        print("Training complete")

    def k_closest_heroes(self, hero_name, k):
        #Returns the k most similar heroes to the input hero_name according to their cosine similarity,
        #NOT according to euclidean distance
        name_to_cossim = {}
        target_hero = np.zeros((1,))
        target_hero[0] = self.name_to_nn[hero_name]

        for nn, hero in self.nn_to_name.items():
            adjacent_hero = np.zeros((1,))
            adjacent_hero[0] = nn
            name_to_cossim[hero] = self.cos_sim_model.predict(
                (target_hero, adjacent_hero))
        return sorted(name_to_cossim, key=name_to_cossim.get, reverse=True)[:k]

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

    def get_hero_embedding(self, hero_nn_number):
        embed_vec = self.compiled_network.layers[2].get_weights()[0][hero_nn_number-1,:]
        return embed_vec

    def visualise_embedding(self):
        # Outputs a 2D t-SNE visualisation of the hero embedding
        embedding = np.zeros(size=(len(self.heroes_df), self.embedding_dim))

        for i in range(len(self.heroes_df)):
            embedding[i,:] = self.get_hero_embedding(hero_nn_number=(i+1))
        embedding_tsne = TSNE(n_components=2).fit_transform(embedding)
        x = embedding_tsne[:, 0]
        y = embedding_tsne[:, 1]
        plt.figure(figsize=(50, 25))
        plt.scatter(x, y)

        for i, name in self.nn_to_name.items():
            plt.annotate(name, (x[i], y[i]))
        plt.show()