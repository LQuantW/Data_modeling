# -*- coding:utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pickle
from datasets import Dataset
from gpfy.likelihoods import Gaussian
from gpfy.model import GP
from gpfy.optimization import create_training_step
from gpfy.spherical import NTK
from gpfy.spherical_harmonics import SphericalHarmonics
from gpfy.variational import VariationalDistributionTriL

key = jax.random.PRNGKey(42)

k = NTK(depth=10)
sh = SphericalHarmonics(num_frequencies=10, phase_truncation=30)
lik = Gaussian()
q = VariationalDistributionTriL()
m = GP(kernel=k)
m_new = m.conditional(sh, q)

param = m_new.init(key,
                   input_dim=32,
                   num_independent_processes=1,
                   likelihood=lik,
                   sh_features=sh,
                   variational_dist=q)
param.__dataclass_fields__
param.params
param = param.set_trainable(collection=k.name, variance=False)
param = param.set_trainable(collection=k.name, variance=True)

# 加载归一化系数
with open('x_mean.pkl', 'rb') as file:
    x_mean = pickle.load(file)
with open('x_std.pkl', 'rb') as file:
    x_std = pickle.load(file)
with open('y_mean.pkl', 'rb') as file:
    y_mean = pickle.load(file)
with open('y_std.pkl', 'rb') as file:
    y_std = pickle.load(file)

# 加载训练好的模型参数
with open('param_new_1000.pkl', 'rb') as file:
    param_new = pickle.load(file)

########## 预测方法 ##########
def predict(x_test):
    '''
    :param x_test: DataFrame数据的一行[C, F1, F2,..., F32]
    :return: numpy.float64
    '''
    x_test = x_test[[f'F{i}' for i in range(1, 33)]]
    # del x_test['C']
    # x_test  = x_test.iloc[0:len(x_test)]
    #x_test = ((x_test['F1':'F32'].values - x_mean) / x_std)[:, None]  # 归一化
    x_test = ((x_test.values - x_mean) / x_std) # 归一化
    test_set = {'x': x_test}
    test_set = Dataset.from_dict(test_set).with_format(type='jax', dtype=jnp.float64)

   
   # print(test_set['x'])
    print(test_set['x'].shape)
    y_pred, _ = m_new.predict_diag(param_new, test_set['x'])
   # print(y_pred)
    return np.array(y_pred) * y_std + y_mean

df = pd.read_csv('./test_example_500.csv')  # 读取测试数据集csv文件
y_pred = predict(df)

print(y_pred)
