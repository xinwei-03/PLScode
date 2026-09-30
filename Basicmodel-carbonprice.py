#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
from datetime import datetime
from statsmodels import regression
import statsmodels.api as sm
import math
import statsmodels.formula.api as smf


# In[ ]:


CP=pd.read_excel(r"C:\Users\\96142\\OneDrive\\Desktop\\论文\\数据\\深圳碳价.xlsx")
SZVIX=pd.read_excel(r"C:\\Users\\96142\\OneDrive\\Desktop\\论文\数据\\VIX中国.xlsx")
BEPU=pd.read_excel(r"C:\Users\96142\OneDrive\Desktop\论文\\数据\\BAKEREPU.xlsx")
DEPU=pd.read_excel(r"C:\\Users\\96142\\OneDrive\\Desktop\\论文\\数据\\DAVIS.xlsx")
IR=pd.read_excel(r"C:\\Users\\96142\\OneDrive\\Desktop\\论文\\数据\\工业增长.xlsx")
GPR=pd.read_excel(r"C:\\Users\\96142\\OneDrive\\Desktop\\论文\\数据\\地缘政治风险.xlsx")
JEU=pd.read_excel(r"C:\\Users\\96142\\OneDrive\\Desktop\\论文\\数据\\JLN - 副本 (2).xlsx")


# In[ ]:


cp_1=CP.iloc[:,2]


# In[ ]:


e_1=SZVIX.iloc[:,0]
e_2=BEPU.iloc[:,2]
e_3=DEPU.iloc[:,2]
e_4=GPR.iloc[:,0]
e_5=JEU.iloc[:,0]
e_6=IR.iloc[:,0]


# In[ ]:


e_1s=[]
e_2s=[]
e_3s=[]
e_4s=[]
e_5s=[]
e_6s=[]


# In[ ]:


for i in range(len(e_1)):
    e_1s.append((e_1[i]-np.mean(e_1))/np.std(e_1))
    e_2s.append((e_2[i]-np.mean(e_2))/np.std(e_2))
    e_3s.append((e_3[i]-np.mean(e_3))/np.std(e_3))
    e_4s.append((e_4[i]-np.mean(e_4))/np.std(e_4))
    e_5s.append((e_5[i]-np.mean(e_5))/np.std(e_5))
    e_6s.append((e_6[i]-np.mean(e_6))/np.std(e_6))
    
Y_1=e_1s
Y_2=e_2s
Y_3=e_3s
Y_4=e_4s
Y_5=e_5s
Y_6=e_6s


# In[ ]:


mod_cp_1=regression.linear_model.OLS(Y_1, cp_1).fit()
mod_cp_2=regression.linear_model.OLS(Y_2, cp_1).fit()
mod_cp_3=regression.linear_model.OLS(Y_3, cp_1).fit()
mod_cp_4=regression.linear_model.OLS(Y_4, cp_1).fit()
mod_cp_5=regression.linear_model.OLS(Y_5, cp_1).fit()
mod_cp_6=regression.linear_model.OLS(Y_6, cp_1).fit()

e_1s=mod_cp_1.resid
e_2s=mod_cp_2.resid
e_3s=mod_cp_3.resid
e_4s=mod_cp_4.resid
e_5s=mod_cp_5.resid
e_6s=mod_cp_6.resid


# In[ ]:


#pls
X=cp_1

X=sm.add_constant(X)
Y_1=e_1s
Y_2=e_2s
Y_3=e_3s
Y_4=e_4s
Y_5=e_5s
Y_6=e_6s

modb = regression.linear_model.OLS(Y_1, X).fit(cov_type='HAC',cov_kwds={'maxlags':5},use_t=True)
eu_1=modb.params[1]
modc = regression.linear_model.OLS(Y_2, X).fit(cov_type='HAC',cov_kwds={'maxlags':5},use_t=True)
eu_2=modc.params[1]
mode = regression.linear_model.OLS(Y_3, X).fit(cov_type='HAC',cov_kwds={'maxlags':5},use_t=True)
eu_3=mode.params[1]
modg = regression.linear_model.OLS(Y_4, X).fit(cov_type='HAC',cov_kwds={'maxlags':5},use_t=True)
eu_4=modg.params[1]
modi = regression.linear_model.OLS(Y_5, X).fit(cov_type='HAC',cov_kwds={'maxlags':5},use_t=True)
eu_5=modi.params[1]
modj = regression.linear_model.OLS(Y_6, X).fit(cov_type='HAC',cov_kwds={'maxlags':5},use_t=True)
eu_6=modj.params[1]


eu_f=[eu_1,eu_2,eu_3,eu_4,eu_5,eu_6]

spls=[]
for i in range(0,len(e_1s)):
    X=np.array(eu_f)
    X=sm.add_constant(X)
    #Y=[Y_1[i],Y_2[i],Y_3[i],Y_5[i],Y_7[i],Y_9[i],Y]
    #Y=[Y_3[i],Y_8[i],Y_7[i],Y_9[i],Y_10[i]]
    #Y=[Y_2[i],Y_3[i],Y_5[i],Y_7[i],Y_9[i],Y_10[i]]
    Y=[Y_1[i],Y_2[i],Y_3[i],Y_4[i],Y_5[i],Y_6[i]]
    Y=np.array(Y)
    mod_first = regression.linear_model.OLS(Y, X).fit()
    spls.append(mod_first.params[1])
    
Y=cp_1
X=spls
X=sm.add_constant(X)
mod_coff = regression.linear_model.OLS(Y, X).fit()
print(mod_coff.summary())


# In[ ]:


#SPCA
Y=cp_1
X_1=e_1s
X_2=e_2s
X_3=e_3s
X_4=e_4s
X_5=e_5s
X_6=e_6s

P_1=[]
P_2=[]
P_3=[]
P_4=[]
P_5=[]
P_6=[]

X_1=sm.add_constant(X_1)
X_2=sm.add_constant(X_2)
X_3=sm.add_constant(X_3)
X_4=sm.add_constant(X_4)
X_5=sm.add_constant(X_5)
X_6=sm.add_constant(X_6)

mod_1 = regression.linear_model.OLS(Y, X_1).fit()
mod_2 = regression.linear_model.OLS(Y, X_2).fit()
mod_3 = regression.linear_model.OLS(Y, X_3).fit()
mod_4 = regression.linear_model.OLS(Y, X_4).fit()
mod_5 = regression.linear_model.OLS(Y, X_5).fit()
mod_6 = regression.linear_model.OLS(Y, X_6).fit()

for i in range(len(X_2)):
    P_1.append(e_1s[i]*mod_1.params[0])
for i in range(len(X_2)):
    P_2.append(e_2s[i]*mod_2.params[0])
for i in range(len(X_2)):
    P_3.append(e_3s[i]*mod_3.params[0])
for i in range(len(X_2)):
    P_4.append(e_4s[i]*mod_4.params[0])
for i in range(len(X_2)):
    P_5.append(e_5s[i]*mod_5.params[0])
for i in range(len(X_2)):
    P_6.append(e_6s[i]*mod_6.params[0])


# In[ ]:


eun=[np.array(P_1),np.array(P_2),np.array(P_3),
           np.array(P_4),np.array(P_5),np.array(P_6)]
eun=np.array(eun)
eun=eun.transpose()

import numpy as np
from numpy.linalg import eig
def pca(X,q):
    X = X - X.mean(axis = 0) #向量X去中心化
    X_cov = np.cov(X.T, ddof = 0) #计算向量X的协方差矩阵，自由度可以选择0或1
    eigenvalues,eigenvectors = eig(X_cov) #计算协方差矩阵的特征值和特征向量
    klarge_index = eigenvalues.argsort()[-q:][::-1] #选取最大的K个特征值及其特征向量
    k_eigenvectors = eigenvectors[klarge_index] #用X与特征向量相乘
    return np.dot(X, k_eigenvectors.T)
X = eun
q = 1
sPCA = pca(X, q)
spca=[]
for i in range(len(X_2)):
    spca.append(sPCA[i][0])

Y=cp_1
X=spca
X=sm.add_constant(X)
mod_coff = regression.linear_model.OLS(Y, X).fit()
print(mod_coff.summary())


# In[ ]:


#PCA
data=pd.read_excel(r"C:\Users\96142\OneDrive\Desktop\论文\数据\工作簿1.xlsx",header=0)

zscore=preprocessing.StandardScaler()
data_zs=zscore.fit_transform(data)
newdata=pd.DataFrame(data_zs)

pca=PCA(n_components=0.95,copy=True)
pca.fit(newdata)

X_pca=pca.fit_transform(newdata)

r=[]
for i in range(len(X_pca)):
   r.append(X_pca[i][0])
save_result=pd.DataFrame({'result':r})

Y=cp_1
X=save_result
X=sm.add_constant(X)
mod_coff = regression.linear_model.OLS(Y, X).fit()
print(mod_coff.summary())

save_result.to_csv(r'C:\Users\96142\OneDrive\Desktop\论文\数据\新建文件夹\PCARES.csv',encoding='utf-8')


# In[ ]:


x=range(len(spca))
spca_pic=list(reversed(spca))
spls_pic=list(reversed(spls))
y_1=spca_pic
y_2=spls_pic
plt.figure(figsize=(16,8))
plt.plot(x,y_2,c='red', label='Economic uncertainty index based on the PLS method')
plt.legend()
plt.savefig('C:\\Users\\96142\\OneDrive\\Desktop\\论文\\pls_pic.png')


# In[ ]:


x=range(len(spca))
spca_pic=list(reversed(spca))
spls_pic=list(reversed(spls))
y_1=spca_pic
y_2=spls_pic
plt.figure(figsize=(16,8))
plt.plot(x,y_1,c='blue', label='Economic uncertainty index based on the scaled PCA method')
#plt.plot(x,y_2,c='red', label='Investor sentiment based on the PLS method')
plt.legend()
plt.savefig('C:\\Users\\96142\\OneDrive\\Desktop\\论文\\spca_pic.png')


# In[ ]:


# ofs
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # 导入归一化工具

def oos_test(y_test, y_pred_model, y_pred_benchmark):
    """
    执行Out-of-Sample检验，计算R²OS、调整后的MSFE及p值。
    """
    # 计算预测误差
    e_model = y_test - y_pred_model
    e_bench = y_test - y_pred_benchmark
    
    # 计算MSFE
    msfe_model = np.mean(e_model ** 2)
    msfe_bench = np.mean(e_bench ** 2)
    
    # R²OS计算
    r2os = 1 - (msfe_model / msfe_bench)
    
    # Clark-West调整统计量计算
    adj_diff = (y_pred_benchmark - y_pred_model) ** 2
    f_t = e_bench ** 2 - (e_model ** 2 - adj_diff)
    mean_f = np.mean(f_t)
    
    # 计算标准误和t统计量
    n = len(f_t)
    std_f = np.std(f_t, ddof=1)  # 无偏标准差
    se_f = std_f / np.sqrt(n)
    t_stat = mean_f / se_f
    
    # 计算单边p值
    p_value = 1 - norm.cdf(t_stat)
    
    return {
        'R2OS': r2os,
        'MSFE_adj': mean_f,
        'p_value': p_value
    }

# 假设 spca 和 cp_1 是原始数据（特征和目标）
X_train, X_test, y_train, y_test = train_test_split(save_result, cp_1, test_size=0.7, shuffle=False)

# 转换为 numpy 数组并修复形状
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)  # 将 y_train 转换为 numpy 数组
y_test = np.array(y_test)    # 将 y_test 转换为 numpy 数组

if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)
if X_test.ndim == 1:
    X_test = X_test.reshape(-1, 1)

# ------------------ 归一化代码 ------------------
# 归一化特征数据 X
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)  # 训练集拟合并转换
X_test_scaled = scaler_X.transform(X_test)        # 测试集仅转换

# 归一化目标数据 y
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()  # 训练集拟合并转换
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()        # 测试集仅转换

# 训练模型（使用归一化后的特征和目标）
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)              # 注意：这里使用 X_train_scaled 和 y_train_scaled
y_pred_model_scaled = model.predict(X_test_scaled)     # 注意：这里使用 X_test_scaled

# 基准模型（历史均值：使用归一化后的 y_train 的均值）
benchmark_pred_scaled = np.mean(y_train_scaled)
y_pred_benchmark_scaled = np.full_like(y_test_scaled, benchmark_pred_scaled)

# 执行OOS检验（使用归一化后的目标数据）
results = oos_test(y_test_scaled, y_pred_model_scaled, y_pred_benchmark_scaled)
print("R²OS:", results['R2OS'])
print("MSFE调整值:", results['MSFE_adj'])
print("p值:", results['p_value'])

