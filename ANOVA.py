import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from statsmodels.formula.api import ols

# 设置文件路径
# data_path = "D://Spring 2024//quant//train//train//"
# file_pattern = "data.{}.csv"
data_path = "../train/train/"
file_pattern = "data.{}.csv"

# 准备数据
def load_data(path, pattern, start, end):
    data_list = []
    source_list = []
    for i in range(start, end + 1):
        file_name = os.path.join(path, pattern.format(i))
        data = pd.read_csv(file_name)
        data_list.append(data)
        source_list.extend([i] * len(data))
    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data, source_list

df, source_list = load_data(data_path, file_pattern, 0, 69)

# 计算每个类别的频数分布
print(df['C'].value_counts())

# 计算每个类别对应的目标变量Y的均值和标准差
category_stats = df.groupby('C')['Y'].agg(['mean', 'std'])
print(category_stats)

'''
df['C'] = df['C'].astype('category')
model = ols('Y ~ C', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
'''

model = ols('Y ~ F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8 + F9 + F10 + F11 + F12 + F13 + F14 + F15 + F16 + F17 + F18 + F19 + F20 + F21 + F22 + F23 + F24 + F25 + F26 + F27 + F28 + F29 + F30 + F31 + F32', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
