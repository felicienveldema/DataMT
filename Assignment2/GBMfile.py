import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import tseries
from pathlib import Path
from scipy import stats
import sklearn
import time
import pickle
from sklearn import ensemble

def loadtraining():
    #LOAD The data
    path = 'Data Mining VU data/training_set_VU_DM_2014.csv'
    df = pd.read_csv(path, sep=',')

    Yclick = df['click_bool']
    Ybook = df['booking_bool']

    X = df.drop(['click_bool', 'booking_bool', 'date_time', 'gross_bookings_usd', 'position'], axis=1)
    #dumb clean NA features
    X = X.fillna(0,axis=1)

    return [X, Yclick, Ybook]

def loadtest():
    #LOAD The data
    path2 = 'Data Mining VU data/test_set_VU_DM_2014.csv'
    dftest = pd.read_csv(path2, sep=',')

    #dumb clean NA features
    Xtest = dftest.fillna(0,axis=1)
    Xtest = Xtest.drop(['date_time'], axis=1)
    return Xtest




def main():
    createGBM = False
    fitGBM = False

    if(createGBM):
        print('create GBM')
        GBM = sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                                    n_estimators=100, subsample=1.0, criterion='friedman_mse',
                                                    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                                    max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None,
                                                    init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None,
                                                    warm_start=False, presort='auto')
        pickle.dump(GBM, open('clean_model.pkl',"wb"))
    else:
        GBM = pickle.load(open('clean_model.pkl', 'rb'))


    print('load training')
    [X, Yclick, Ybook] = loadtraining()
    Xtrain = X[:100000]


    Yclick = Yclick + Ybook

    print('fit')
    start = time.time()
    if(fitGBM):
        GBM.fit(Xtrain, Yclick[:100000])
        pickle.dump(GBM, open('fitted_model.pkl', 'wb'))
    else:
        GBM = pickle.load(open('fitted_model.pkl', 'rb'))
    end = time.time()
    print(end - start)

    print('load test')
    start = time.time()
    #Xtest = loadtest()
    Xtest = X[100000:190000]
    end = time.time()
    print(end - start)

    print('Scoring')
    # start = time.time()
    # Y = GBM.predict(Xtest)
    # end = time.time()
    # print(end - start)

    print(Yclick[:4])
    print(GBM.score(Xtest, Yclick[100000:190000]))

    Y = GBM.predict(Xtest)
    print(sum(Y))


if __name__ == '__main__':
    main()