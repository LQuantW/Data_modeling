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
# print(m)
m_new = m.conditional(sh, q)
# print(m_new)

# 加载训练数据集
print('加载训练数据集...')
num_features = 32  # 特征个数
x_train = np.zeros((0, num_features))  # 训练输入
y_train = np.zeros((0, 1))  # 训练输出
for i in range(70):
    df = pd.read_csv('./train/data.' + str(i) + '.csv')
    x_train = np.concatenate((x_train, df.loc[:, 'F1':'F32'].values), axis=0)
    y_train = np.concatenate((y_train, df.loc[:, 'Y':'Y'].values), axis=0)
print(np.sum(np.abs(y_train) > 0.3))

# 归一化
x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
y_mean, y_std = np.mean(y_train), np.std(y_train)
print('y_mean =', y_mean)
print('y_std =', y_std)
print('y_max =', np.max(np.abs(y_train)))
x_train = (x_train - x_mean) / x_std
y_train = (y_train - y_mean) / y_std

data_dict = {'x': x_train, 'y': y_train}
dataset = Dataset.from_dict(data_dict).with_format(type='jax', dtype=jnp.float64)
# dataset = dataset.train_test_split(test_size=0.9)

param = m_new.init(key,
                   input_dim=x_train.shape[-1],
                   num_independent_processes=y_train.shape[-1],
                   likelihood=lik,
                   sh_features=sh,
                   variational_dist=q)
param.__dataclass_fields__
param.params
print(param._trainables['NTK']['variance'])
param = param.set_trainable(collection=k.name, variance=False)
print(param._trainables['NTK']['variance'])
param = param.set_trainable(collection=k.name, variance=True)
print(param._trainables['NTK']['variance'])

train_step = create_training_step(m_new, dataset, ('x', 'y'), q, lik)
param_new, state, elbos = m_new.fit(param, train_step, optax.adam(2e-2), num_iters=1000)
# 1000：1451565.68，0.2088052293508016
mu_pred, _ = m_new.predict_diag(param_new, dataset['x'])
print(jnp.sqrt(jnp.mean((dataset['y'] - mu_pred) ** 2)) * y_std)

with open('./param_new_1000.pkl', 'wb') as file:
    pickle.dump(param_new, file)
