import numpy as np
import pandas as pd
import json
import requests

all_heroes_json = requests.get('https://api.opendota.com/api/heroes').json()
all_heroes_df = pd.DataFrame(all_heroes_json)
print(all_heroes_df.head(10))

roles = []
for role in all_heroes_df['roles']:
    roles = roles + role
    
roles = list(set(roles))
print(roles)
for role in roles:
    all_heroes_df[role] = all_heroes_df['roles'].apply(lambda x: role in x)

print(all_heroes_df.head())

all_heroes_df = all_heroes_df.drop("roles",axis=1)
all_heroes_df.to_csv("heroes.csv",index=False)