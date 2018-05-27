import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
import pickle
da=pd.read_csv('test1.csv', sep=',')
x_test=da[['prop_starrating', 'price_usd','prop_review_score', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2','random_bool', 'click_bool']].values
y_test=da[['booking_bool']].values

y_test=y_test.reshape(-1)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf=pickle.load(open('rf_click_book.sav', 'rb'))
errors=abs(rf.predict(x_test) - y_test)
print(errors)
#print(cross_val_score(rf, x_train, y_train, cv=10, scoring='accuracy'))
#print(rf.feature_importances_)
#print(rf.score(x_test, y_test))
print('Mean absolute accuracy rate')
print(1-np.sum(errors)/len(y_test))

