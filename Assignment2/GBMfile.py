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

def loadtraining(path, balanced=False):
    #LOAD The data
    df = pd.read_csv(path)
    print(len(df))
    if(balanced):
        print("Balancing the dataset")
        neg = 4
        ilist = []
        srchid = 0
        for i in range(0,len(df)):
            cur_srchid = df['srch_id'].loc[i]
            if (df['click_bool'].loc[i] == 1 | df['booking_bool'].loc[i] == 1):
                neg += 2
            elif(df['click_bool'].loc[i] == 0 & df['booking_bool'].loc[i] == 0):
                if(neg > 0):
                    srchid = cur_srchid
                    neg += -1
                if(neg <= 0):
                    ilist.append(i)
            if (i % 50000 == 0 ):
                print(i)
            #     if  i > 100:
        df = df.drop(index=ilist)
        # df.to_csv('balancedtrain.csv')
        # print('Wrote:  ', i)
        print(len(df))
        print("writing")
        df.to_csv('balancedtrain2.csv')
        print("writing complete")

    Yclick = df['click_bool']
    Ybook = df['booking_bool']



    X = df.drop(['click_bool', 'booking_bool', 'position'], axis=1)
    #dumb clean NA features
    X = X.fillna(0,axis=1)

    return [X, Yclick, Ybook]

def loadtest():
    #LOAD The data
    path2 = 'Data Mining VU data/test.csv'
    dftest = pd.read_csv(path2, sep=',')

    #dumb clean NA features
    Xtest = dftest.fillna(0,axis=1)
    # Xtest = Xtest.drop(['date_time'], axis=1)
    return Xtest




def main():
    createGBM = False

    #Create the balanced dataset or not
    balanced= False

    #train the dataset
    fitGBM = True
    writeGBM = True

    if(createGBM):
        print('create GBM')
        GBM = sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                                    n_estimators=600, subsample=1.0, criterion='friedman_mse',
                                                    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.5,
                                                    max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None,
                                                    init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None,
                                                    warm_start=False, presort='auto')
        pickle.dump(GBM, open('clean_model_ne600_learn0_1.pkl',"wb"))
    else:
        GBM = pickle.load(open('clean_model_ne600_learn0_1.pkl', 'rb'))


    print('load training')
    #change location to train
    # [Xtrain, Yclick, Ybook] = loadtraining('Data Mining VU data/train.csv', balanced)
    [Xtrain, Yclick, Ybook] = loadtraining('balancedtrain.csv', balanced)
    Ytrain = Yclick + Ybook

    print('fit')
    start = time.time()
    if(fitGBM):
        sample_weight = np.abs(np.random.randn(len(Ytrain)))
        GBM.fit(Xtrain, Ytrain, sample_weight)
        pickle.dump(GBM, open('fitted_model_ne600_learn0_1.pkl', 'wb'))
    else:
        GBM = pickle.load(open('fitted_model_ne600_learn0_1.pkl', 'rb'))
    end = time.time()
    print(end - start)

    print('load test')
    start = time.time()
    [Xtest, Yclicktest, Ybooktest] = loadtraining('Data Mining VU data/test1.csv')
    end = time.time()
    print(end - start)

    print('Scoring')
    Ytest = Yclicktest + Ybooktest
    print(GBM.score(Xtest, Ytest))

    Y = GBM.predict(Xtest)
    print(sum(Ytest))
    print(sum(Y))

    #prediction to file:
    if(writeGBM):
        file =  pd.DataFrame(columns=['srch_id', 'prop_id', 'pred_click', 'pred_book'])
        index = 0
        for i, content in enumerate(Y):
            if content == 2:
                data = Xtest[['srch_id', 'prop_id']].values[i]
                file.loc[index] = np.append(data,[1, 1])
                index += 1

            if content == 1:
                data = Xtest[['srch_id', 'prop_id']].values[i]
                file.loc[index] = np.append(data,[1, 0])
                index += 1
            ##Turn on to add zero cases
            # if content == 0:
            #     data = Xtest[['srch_id', 'prop_id']].values[i]
            #     file.loc[index] = np.append(data,[0, 0])
            #     index += 1

        file.to_csv(path_or_buf='GBM_ne600_learn0_1.csv', sep=',')

    importances = GBM.feature_importances_
    features = ('prop_starrating', 'price_usd', 'prop_review_score', 'prop_location_score1', 'prop_location_score2',
                'prop_log_historical_price', 'promotion_flag', 'srch_query_affinity_score', 'cr1', 'cr2', 'random_bool')
    indices = np.argsort(importances)
    plt.title('Feature Importances-GBM model')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), features)
    plt.xlabel('Relative Importance')
    plt.show()

if __name__ == '__main__':
    main()