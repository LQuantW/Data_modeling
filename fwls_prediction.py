import pandas as pd
import matplotlib
from scipy import stats
matplotlib.use('TkAgg')
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt
import os
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan #BP检验
import  numpy as np
from sklearn.model_selection import train_test_split
import sys 
from sklearn.metrics import mean_squared_error



def prediction(data_example, X_test):
    X_list = '+'.join(data_example.columns[1:-1])
    # OLS:
    reg_ols = smf.ols(formula='Y ~ ' + X_list, data=data_example)
    results_ols = reg_ols.fit()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    


    # log(残差平方)对自变量做回归
    data_example['loge2'] = np.log(results_ols.resid ** 2)

    data_train, data_test = train_test_split(data_example, test_size=0.2, random_state=0)

    # reg_loge2 = smf.ols(formula='loge2 ~ ' + X_list, data=data_example)
    reg_loge2 = smf.ols(formula='loge2 ~ ' + X_list, data=data_train)

    results_loge2 = reg_loge2.fit()

    # FWLS
    wls_weight = list(1 / np.exp(results_loge2.fittedvalues))
    # reg_wls = smf.wls(formula='Y ~ ' + X_list, weights=wls_weight, data=data_example)
    reg_wls = smf.wls(formula='Y ~ ' + X_list, weights=wls_weight, data=data_train)

    

    results_wls = reg_wls.fit()

    print(data_test)

    data_test_x = data_test.drop(columns = {'Y'})
    data_test_y = data_test['Y']

    
    p_test = results_wls.predict(data_test_x)
    mse = mean_squared_error(p_test, data_test_y)

    print("mse: ", mse)
    print("rmse: ", np.sqrt(mse))



    prediction_value = results_wls.predict(X_test)
    return prediction_value

if __name__ == '__main__':
    root_path = "../"

    # path_data = sys.path[-1]+'/train/train'
    # save_path =sys.path[-1]+ '/Heteroskedasticity'

    path_data = root_path + '/train/train'
    save_path = root_path + '/Heteroskedasticity'

    # file_list = os.listdir(data_path)
    file = 'train_data'
    data_example = pd.DataFrame([])
    # X_test = pd.read_csv(sys.path[-1]+'/test_example.csv')
    X_test = pd.read_csv(root_path+'train/test_example.csv')
    for i in range(30):
        df = pd.read_csv(path_data+'/'+'data.'+str(i)+'.csv')
        data_example = pd.concat([data_example,df])


    data_example = data_example.drop(columns = {'C'})

    X_test = X_test.drop(columns = {'C'})
    res = prediction(data_example=data_example, X_test=X_test)
    print(res)





