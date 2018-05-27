import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('Data Mining VU data/training_set_VU_DM_2014.csv', sep=',')
df2 = pd.read_csv('train_data.csv', sep=',')

df2['srch_id']=df['srch_id']
df2['prop_id']=df['prop_id']

x=df2[['srch_id','prop_id','prop_starrating', 'price_usd','prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2','random_bool','click_bool', 'position', 'booking_bool']]

x_train, x_test= train_test_split(x, test_size=0.2, shuffle=False)


x_train.to_csv('train1.csv')
x_test.to_csv('test1.csv')
