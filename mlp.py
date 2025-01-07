# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, learning_curve
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import time
# import matplotlib
# import matplotlib.pyplot as plt
# from sklearn.model_selection import RandomizedSearchCV
# # 设置时间测量
# start_time = time.time()
# matplotlib.use('agg')  # 不显示图形，保存图形文件
#
# # 加载数据集
# data = pd.read_excel('C:/Users/a1830/Desktop/模型/机器学习/output_file.xlsx')#"C:\Users\a1830\Desktop\模型\机器学习\output_file.xlsx"
# # 如果有日期列，先移除或转换日期列
# if '日期' in data.columns:  # 替换为你的日期列名
#     # 移除日期列（如果不需要）
#     data = data.drop(columns=['日期'])
#
#     # 或者将日期转换为其他格式，例如年、月等
#     # data['year'] = pd.to_datetime(data['date_column']).dt.year
#     # data['month'] = pd.to_datetime(data['date_column']).dt.month
#     # data = data.drop(columns=['date_column'])
#
# # 分割特征和目标变量
# X = data.drop('野生羊肚菌干产量/Kg', axis=1)  # 预测目标列
# y = data['野生羊肚菌干产量/Kg']
# # 删除包含 NaN 的样本
# X = X.dropna()
# y = y[X.index]  # 保持与 X 对应的标签
#
# # 对数据进行归一化
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 建立MLP回归模型
# mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, learning_rate_init=0.001, random_state=42)
# mlp.fit(X_train, y_train)
#
#
#
# # 训练模型
# mlp.fit(X_train, y_train)
#
# # 预测
# y_pred = mlp.predict(X_test)
#
# # 计算均方误差和R²分数
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error: {mse:.2f}")
# print(f'R^2 Score: {r2:.2f}')
#
#
# # 保存预测结果
# pd.Series(y_pred).to_csv('mlp_regression_result.csv', index=False)
#
# # 绘制回归曲线：真实值 vs 预测值
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.values, label='True Values', color='blue', marker='o')
# plt.plot(y_pred, label='Predicted Values', color='red', marker='x')
# plt.xlabel('Samples')
# plt.ylabel('Values')
# plt.title('MLP Regression: True vs Predicted Values')
# plt.legend()
# plt.savefig('mlp_regression_curve.png')  # 保存回归曲线图
#
# # 绘制学习曲线
# train_sizes, train_scores, test_scores = learning_curve(mlp, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=2)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.figure(figsize=(12, 8))
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.savefig('mlp_learning_curve.png')  # 保存学习曲线图
#
# # 输出程序运行时间
# end_time = time.time()
# print("Running time:", end_time - start_time)
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import matplotlib
import matplotlib.pyplot as plt

# 设置时间测量
start_time = time.time()
matplotlib.use('agg')  # 不显示图形，保存图形文件

# 加载数据集
data = pd.read_json('/mnt/random_forest/task1/data1.jsonl', lines=True)

# 如果有日期列，先移除或转换日期列
if '日期' in data.columns:  # 替换为你的日期列名
    data = data.drop(columns=['日期'])

# 分割特征和目标变量
X = data.drop('野生羊肚菌干产量/Kg', axis=1)  # 预测目标列
y = data['野生羊肚菌干产量/Kg']

# 删除包含 NaN 的样本
X = X.dropna()
y = y[X.index]  # 保持与 X 对应的标签

# 对数据进行归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立MLP回归模型
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=20000, learning_rate_init=0.004, random_state=42, tol=1e-4)

mlp.fit(X_train, y_train)

# 预测
y_pred = mlp.predict(X_test)
y_pred = y_pred.round(2)
# 计算均方误差和R²分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f'R^2 Score: {r2:.2f}')

# 保存预测结果
pd.Series(y_pred).to_csv('mlp_regression_result.csv', index=False)

# 绘制回归曲线：真实值 vs 预测值
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Values', color='blue', marker='o')
plt.plot(y_pred, label='Predicted Values', color='red', marker='x')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.title('MLP Regression: True vs Predicted Values')
plt.legend()
plt.savefig('mlp_regression_curve.png')  # 保存回归曲线图

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(mlp, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=2)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(12, 8))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.savefig('mlp_learning_curve.png')  # 保存学习曲线图

# 输出程序运行时间
end_time = time.time()
print("Running time:", end_time - start_time)


