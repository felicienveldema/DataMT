import numpy as np
import pandas as pd

selected_columns = ['visitor_location_country_id','prop_country_id', 'prop_starrating', \
                    'prop_review_score', 'prop_location_score1', 'prop_location_score2', \
                    'prop_log_historical_price', 'position', 'price_usd','promotion_flag', \
                    'srch_query_affinity_score', 'random_bool', 'click_bool', 'booking_bool', \
                    'cr1','cr2']



# All attributes we want select
selected_columns = ['visitor_location_country_id','prop_country_id', 'prop_starrating', \
                    'prop_review_score', 'prop_location_score1', 'prop_location_score2', \
                    'prop_log_historical_price', 'price_usd','promotion_flag', \
                    'srch_query_affinity_score', 'random_bool']

# The attributes on which we are going to standardize on
# -> So for any datapoint x, we want to standardize x with respect to all datapoints with same
# 'visitor_location_country_id' AND 'prop_country_id' as x
standardize_on = ['visitor_location_country_id', 'prop_country_id']

# Of the selected attributes, the attributes which we standardize
columns_to_standardize = ['prop_starrating','price_usd', 'prop_review_score', \
                          'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price']

# Read whole dataset
df = pd.read_csv('pcadf.csv', sep=',')

df_selection = df[selected_columns]
df_selection = df_selection.fillna(df_selection.median())

# For each "key" ('visitor_location_country_id', 'prop_country_id'), we
# standardize a numeric attribute as a-> (a-m)/s, m=mean,s=std
# The dict "transf_param" remembers for each key encountered in whole training set the mean and std
# as tuple (m,s) where m,s both numpy arrays

transf_param = {}
k = 0
for name,group in df_selection[standardize_on+columns_to_standardize].groupby(by=standardize_on):
    mean = np.array(group[columns_to_standardize].mean())
    #print("mean is: " + str(mean))
    std = np.array(group[columns_to_standardize].std())
    #print("std is: " + str(std))

    if True in np.isnan(mean):# or True in np.isnan(std):
        print("mean" + str(name))
    
    if True in np.isnan(std):
        std = np.ones(len(columns_to_standardize))
        print("std " + str(name))
#        print(group)

    # If std gives 0, replace with 1
    epsilon = 0.01            # Reason is that we have found some cases where all values are 0.69, but std wasnt 0
    std[std<epsilon] = 1
    
    transf_param[name] = (mean,std)
    k=k+1
    if k % 18000 == 0:
        print(k)
        break


# Save the dict as file.
np.save('transf_param.npy', transf_param)

# Load dict from file
saved_transf_param = np.load('transf_param.npy').item()
# Load whole test data
df_test = pd.read_csv('Data Mining VU data/test_set_VU_DM_2014.csv', sep=',')
print(df_test.shape)
df_test = df_test[selected_columns]
print(df_test.describe())

num_attr_to_stand = len(columns_to_standardize)
avg_mean = np.zeros(num_attr_to_stand)
avg_std = np.zeros(num_attr_to_stand)

for key,value in saved_transf_param.items():

    avg_mean = avg_mean + value[0]
    avg_std = avg_std + value[1]

avg_mean = avg_mean / len(saved_transf_param)
avg_std = avg_std / len(saved_transf_param)
print(avg_mean)
print(avg_std)

# In the test set, we go through each ('visitor_location_country_id', 'prop_country_id') group
# and standardize the values of (some of ) the attributes

# Remember howmany groups in total, and howmany of them were not encountered before
num_data_points = 0
num_data_points_unknown = 0
num_unknowns = 0
k = 0
for name,group in df_test.groupby(by=standardize_on):
    if k % 6000 == 0:
        print(k)
    k = k+1
    num_data_points = num_data_points + group.shape[0]
    
    if name in saved_transf_param:
        # Means we have encountered the key in training set
        cur_params = saved_transf_param[name]
#        print(cur_params)
    else:
        # Means we havent encountered the key in training set
        print("Unknown group " + str(k) + ", with key: " + str(name))
        num_unknowns = num_unknowns + 1
        num_data_points_unknown = num_data_points_unknown + group.shape[0]
        # For unknown key, we have use:
        #  - mean = avg mean of all known keys
        #  - std = avg std of all known keys
        cur_params = (avg_mean*np.ones(num_attr_to_stand),avg_std*np.ones(num_attr_to_stand))

#    print(df_test.loc[(df_test['visitor_location_country_id'] == name[0]) & (df_test['srch_destination_id'] == name[1])])
    df_test.loc[(df_test['visitor_location_country_id'] == name[0]) & (df_test['prop_country_id'] == name[1]), \
                columns_to_standardize] = \
            (df_test.loc[(df_test['visitor_location_country_id'] == name[0]) & \
                        (df_test['prop_country_id'] == name[1]), columns_to_standardize] - cur_params[0]) / cur_params[1]
#    print(df_test.loc[(df_test['visitor_location_country_id'] == name[0]) & (df_test['srch_destination_id'] == name[1])])

print("total num of unknowns is: " + str(num_unknowns))
print("total num groups is: " + str(k))
print("total num of unknown data points is: " + str(num_data_points_unknown))
print("total num of data points is: " + str(num_data_points))


df=df.dropna(subset = ['promotion_flag'])
df=df.dropna(subset = ['random_bool'])

df=df.fillna(0)
pca=pd.read_csv('test_pca.csv', sep=',')
df['cr1']=pca[['cr1']]
df['cr2']=pca[['cr2']]
df.to_csv('test_data.csv')


df_test.to_csv('test_stand.csv')


