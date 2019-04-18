import pandas as pd
import numpy as np
import json as js
import time as tm
import requests as rq

odota_url = "https://api.opendota.com/api/publicMatches?less_than_match_id="
initial_id = 4602282566 #latest match id as of 02/04/2019 19:19
current_id = initial_id
tail_id = 0
cols = ['avg_mmr',
        'avg_rank_tier',
        'cluster',
        'dire_team',
        'duration',
        'game_mode',
        'lobby_type',
        'match_id',
        'match_seq_num',
        'num_mmr',
        'num_rank_tier',
        'radiant_team',
        'radiant_win',
        'start_time']
dota_matches = pd.DataFrame(columns=cols)

for i in range(3000):
    try:
        tm.sleep(1.2)
        request_url = odota_url + str(current_id)
        response_json = rq.get(request_url).json()
        response_df = pd.DataFrame.from_dict(response_json, orient='columns')
        dota_matches = dota_matches.append(response_df)
        tail_id = response_df.iloc[-1,:]['match_id']
        current_id = tail_id - 1
        print(str(i) + " is gucci")
    except:
        tm.sleep(1.2)
        print("wops a dops, we failed on " + str(i))

dota_matches.to_csv("dota_matches.csv",index=False)
