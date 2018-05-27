import numpy as np
import pandas as pd

col_selection = ['srch_id', 'prop_id', \
                 'prop_starrating', 'price_usd', 'prop_review_score', \
                 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', \
                 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2', 'random_bool']

df_test_A = pd.read_csv("test_data.csv", sep=',')

df_test_B = pd.read_csv("test_set_VU_DM_2014.csv", sep=',', usecols=[0,7])

df_test_A['srch_id'] = df_test_B['srch_id']
df_test_A['prop_id'] = df_test_B['prop_id']

df_test_A = df_test_A[col_selection]
print(df_test_A)

df_test_A.to_csv("VU_test_data.csv", header=False, index=False)
