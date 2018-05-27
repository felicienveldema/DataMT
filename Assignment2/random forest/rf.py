import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
import pickle

data=pd.read_csv('train.csv', sep=',')
#df= pd.read_csv('Data Mining VU data/training_set_VU_DM_2014.csv', sep=',')
#x1=df[['visitor_hist_adr_usd','prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff', 'click_bool', 'gross_bookings_usd']].values
#y1=df[['booking_bool']].values
#x1=np.nan_to_num(x1)
#y1=np.nan_to_num(y1)
#y1=y1.reshape(-1)
x_train=data[['prop_starrating', 'price_usd','prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2','random_bool', 'click_bool']].values
y_train=data[['booking_bool']].values
y_train=y_train.reshape(-1)

df=pd.read_csv('test.csv', sep=',')
x_test=df[['prop_starrating', 'price_usd','prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2','random_bool', 'click_bool']].values

y_test=df[['booking_bool']].values
y_test=y_test.reshape(-1)

#Create the random forest regressor for booking
regr = RandomForestClassifier(max_depth=10, random_state=0)
regr.fit(x_train,y_train )

print(regr.feature_importances_)
print(regr.score(x_train, y_train))
print(regr.score(x_test, y_test))
print(np.sum(regr.predict(x_test)))
print(regr.predict(x_test))
print(np.sum(y_test))
print(cross_val_score(regr, x_train, y_train, cv=10, scoring='accuracy'))


filename = 'rf_book_nb.sav'
pickle.dump(regr, open(filename, 'wb'))
