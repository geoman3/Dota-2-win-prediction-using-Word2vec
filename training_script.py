import embedding_handler as eh
import pandas as pd

heroes = pd.read_csv('heroes.csv')
all_matches = pd.read_csv('dota_matches.csv')
ranked_matches = all_matches[all_matches['game_mode'] == 22]
all_teams = list(ranked_matches['radiant_team']) + list(ranked_matches['dire_team'])

for dim in range(4):
    for epo in range(7):
        working_model = eh.EmbeddingNN(heroes_df=heroes,embedding_dim=((2*(dim+1))+12))
        working_model.load_odota_data(all_teams)
        working_model.train_model(num_in_batch=100,num_epochs=(epo+1))
        working_model.save_model(file_name="embeddim"+str(((2*(dim+1))+12))+"epochs"+str((epo+1))+".h5")