import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import tseries
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

df_1 = pd.read_csv('pcadf.csv', sep=',')
df_2=df_1[['visitor_location_country_id','prop_country_id', 'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd','promotion_flag', 'srch_query_affinity_score', 'random_bool', 'click_bool', 'booking_bool','cr1','cr2']]

n_features=df_2[['visitor_location_country_id','prop_country_id', 'prop_starrating','price_usd', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price']]
df_2=df_2.drop(['prop_starrating','price_usd', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price'],axis=1)
n_features=n_features.groupby(['visitor_location_country_id','prop_country_id']).transform(lambda x: (x - x.mean()) / x.std())
n_features['visitor_location_country_id']=pd.Series(df_2['visitor_location_country_id'])
n_features['prop_country_id']=pd.Series(df_2['prop_country_id'])
n_features=n_features.groupby(['visitor_location_country_id','prop_country_id']).transform(lambda x: x.fillna(x.mean()))
#n_features.to_csv('features.csv')
df_3=pd.concat([df_2, n_features], axis=1)
df_3=df_3.dropna(subset = ['promotion_flag'])
df_3=df_3.dropna(subset = ['random_bool'])
df_3=df_3.dropna(subset = ['click_bool'])
df_3=df_3.fillna(0)
df_3.to_csv('n_data.csv')
