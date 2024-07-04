import pandas as pd 
import numpy as np

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc


uscols = ["F"+str(i) for i in range(1,33)]
uscols = ["Y"] + uscols
dfs = []
for j in range(69):
    path = "../yujin/train/train/data.{}.csv".format(j)
    dfs.append(pd.read_csv(path, delimiter=',', usecols=uscols))

print("www")

df_train = pd.concat(dfs)


df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['F17','Y'], axis=1)
y_train = df_train['Y'].values
print(x_train.shape, y_train.shape)




del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)



params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          
params['sub_feature'] = 0.5      
params['bagging_fraction'] = 0.85 
params['bagging_freq'] = 40
params['num_leaves'] = 512        
params['min_data'] = 500         
params['min_hessian'] = 0.05     
params['verbose'] = 0

print("Fitting LightGBM model\n")
#model = lgb.train(params, d_train, 430)
model = lgb.Booster(model_file='lgbm_model.mdl')
#model.save_model('lgbm_model.mdl')

del d_train; gc.collect()
del x_train; gc.collect()


# path = "./train/data.69.csv"
path = "../yujin/train/train/data.69.csv"
df_test = pd.read_csv(path, delimiter=',', usecols=uscols)
x_test = df_test.drop(['F17','Y'], axis=1)
y_test = df_test['Y'].values
p_test = model.predict(x_test)


def predict(x):
    return model.predict(x)

del x_test; gc.collect()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(p_test, y_test)

print("mse: ", mse)
print("rmse: ", np.sqrt(mse))



