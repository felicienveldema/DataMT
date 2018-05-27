import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
import pickle
da=pd.read_csv('test1.csv', sep=',')
rf1=pickle.load(open('rf_click.sav', 'rb'))
rf2=pickle.load(open('rf_book_nb.sav', 'rb'))

x_test=da[['prop_starrating', 'price_usd','prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2','random_bool']].values
click_predict=rf1.predict(x_test)
df2=da[['prop_starrating', 'price_usd','prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2','random_bool']]
df2['click_bool']=click_predict
x1_test=df2[['prop_starrating', 'price_usd','prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2','random_bool','click_bool' ]].values
book_predict=rf2.predict(x1_test)
df2['booking_bool']=book_predict
df2['srch_id']=da[['srch_id']]
df2['prop_id']=da[['prop_id']]
df3=df2[['srch_id', 'prop_id', 'click_bool', 'booking_bool']]
df3.to_csv('rf_out.csv')
