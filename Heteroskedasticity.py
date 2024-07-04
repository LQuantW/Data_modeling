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


def get_reult(res):
    df_out = pd.concat((res.params, res.tvalues, res.conf_int()), axis=1)
    df_out.columns = ['coef', 't', 'CI_low', 'CI_high']
    return df_out

root_path = "../"

# path_data = sys.path[-1]+'\\train\\train'
# save_path =sys.path[-1]+ '\\Heteroskedasticity'
path_data = root_path + 'train/train'
save_path = root_path + 'Heteroskedasticity'

# file_list = os.listdir(data_path)
file = 'train_data'
data_example = pd.DataFrame([])
test_data = pd.read_csv(root_path+'train/test_example.csv')
for i in range(30):
    df = pd.read_csv(path_data+'/'+'data.'+str(i)+'.csv')
    data_example = pd.concat([data_example,df])
# file = file_list[0]

# data_example = pd.read_csv(data_path+'\\'+file)

data_example = data_example.drop(columns = {'C'})
##########################生成残差散点图###############################################
X_col = data_example.columns[1:-1]
X_list = '+'.join(data_example.columns[1:-1])
reg = smf.ols(formula='Y ~ '+X_list, data=data_example)
results = reg.fit()
print(results.summary())



res=results.resid #从OLS回归模型中获取残差
fitted=results.fittedvalues #从OLS回归模型中获取拟合值

#残差与拟合值的散点图
plt.figure(1)
plt.scatter(fitted,res)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Scatter of residuals versus fitted values '+file)
plt.show()
plt.savefig(save_path+'/'+'fig'+'/'+'Scatter of residuals versus fitted values '+file+'.png')
plt.close()


##############################BP检验###########################

data_example['resid_sq'] = results.resid ** 2 #定义残差的平方
reg_resid = smf.ols(formula='resid_sq ~ '+X_list, data=data_example)
results_resid = reg_resid.fit()
bp_F_statistic = results_resid.fvalue
bp_F_pval = results_resid.f_pvalue
print(f'bp检验的F统计量: {bp_F_statistic}')
print(f'bp检验的F统计量对应的p值: {bp_F_pval}')
df_pb=pd.DataFrame.from_dict({'bp检验的F统计量':[bp_F_statistic],'bp检验的F统计量对应的p值':[bp_F_pval]})
df_pb.to_csv(save_path+'/'+'bp_stats'+'/'+file,index=False)
################################WHLS回归########################

#OLS:
reg_ols = smf.ols(formula='Y ~ ' + X_list,data=data_example)
results_ols = reg_ols.fit()
print(results_ols.summary())

df_ols = get_reult(results_ols)
df_ols.to_csv(save_path+'/'+'res_ols'+'/'+file)

#log(残差平方)对自变量做回归
data_example['loge2'] = np.log(results_ols.resid ** 2)
reg_loge2 = smf.ols(formula='loge2 ~ '+ X_list, data=data_example)
results_loge2 = reg_loge2.fit()

# FWLS
wls_weight = list(1 / np.exp(results_loge2.fittedvalues))
reg_wls = smf.wls(formula='Y ~ ' + X_list,weights=wls_weight, data=data_example)
results_wls = reg_wls.fit()
print(results_wls.summary())

df_wls =get_reult(results_wls)

df_wls.to_csv(save_path+'/'+'res_fwls'+'/'+file)

# a=results_wls.(data_example[X_col])


############################交叉检验模型的有效性########################################
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score

train_set,test_set = train_test_split(data_example,test_size=0.5,random_state=1)
train_set =train_set.reset_index(drop=True)
#OLS:
reg_ols_train = smf.ols(formula='Y ~ '+X_list,data=train_set)
results_ols_train = reg_ols_train.fit()

df_ols = get_reult(results_ols_train)
# df_ols.to_csv(save_path+'\\'+'res_ols'+'\\'+file)

# print(results_ols_train.resid)
#
# print(np.log(results_ols_train.resid ** 2))
#log(残差平方)对自变量做回归
train_set['loge2_cross'] = np.log(results_ols_train.resid ** 2)
reg_loge2_train = smf.ols(formula='loge2_cross ~ '+X_list, data=train_set)
results_loge2_train = reg_loge2_train.fit()

# FWLS
wls_weight_train = list(1 / np.exp(results_loge2_train.fittedvalues))
reg_wls_train = smf.wls(formula='Y ~ '+X_list,weights=wls_weight_train, data=train_set)
results_wls_train = reg_wls_train.fit()

prediction_value = results_wls.predict(test_set[X_col])
prediction_value_ols = results_ols_train.predict(test_set[X_col])
score_WFLS= r2_score(prediction_value,test_set['Y'])
score_OLS = r2_score(prediction_value_ols,test_set['Y'])
print(f'WFLS测试效果:{score_WFLS}')
print(f'OLS测试效果:{score_OLS}')

print(results_wls.summary())

df_wls =get_reult(results_wls)

df_wls.to_csv(save_path+'\\'+'res_fwls'+'\\'+file)