# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import shap
#
#
# # 1. 加载 Excel 文件
# file_path = r'C:\Users\a1830\Desktop\模型\1-GBM_dataset.xlsx'#C:\Users\a1830\Desktop\模型\1-GBM_dataset.xlsx\C:/Users/a1830/Desktop/DATA SET.xlsx
#
#  # 使用原始字符串 r'' 处理路径"C:\Users\a1830\Desktop\DATA SET.xlsx"
# df = pd.read_excel(file_path, engine='openpyxl')
#
# # 显示数据的前几行，以确认数据是否正确加载
# print("原始数据：")
# print(df.head().to_string(index=False))
# # 2. 数据预处理
# # 删除第一列（日期）
# df.drop(df.columns[0], axis=1, inplace=True)
#
# # 检查是否有缺失值
# if df.isnull().values.any():
#     print("警告：存在缺失值")
#
# # 将数据分为特征和目标变量
# X = df.iloc[:, :-1]  # 所有的特征
# y = df.iloc[:, -1]   # 目标变量（野生羊肚菌干产量）
#
# # 创建一个简单的填充器和标准化处理器
# imputer = SimpleImputer(strategy='mean')  # 使用平均值填充缺失值
# scaler = StandardScaler()  # 标准化处理器
#
# # 预处理步骤
# preprocessor = Pipeline([
#     ('imputer', imputer),
#     ('scaler', scaler),
# ])
#
# # 对特征数据应用预处理器
# X_preprocessed = preprocessor.fit_transform(X)
#
# # 输出归一化后的数据前几行
# print("归一化后的数据：")
# print(pd.DataFrame(X_preprocessed).head())
#
# # 3. 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
#
# # 4. 创建随机森林回归模型
# rf = RandomForestRegressor(random_state=42)
#
# # 超参数调优
# param_grid = {
#     'n_estimators': [100, 200, 300],  # 决策树数量
#     'max_depth': [10, 20, 30],  # 每棵树的最大深度
#     'min_samples_split': [5, 10, 15],  # 内部节点再次分裂所需最小样本数
#     'min_samples_leaf': [1, 2, 4]  # 叶子节点所需最小样本数
# }
#
# # 创建网格搜索对象
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
#
# # 训练模型
# grid_search.fit(X_train, y_train)
#
# # 输出最佳参数
# print("Best parameters found: ", grid_search.best_params_)
#
# # 使用最佳参数进行预测
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test)
#
# # 5. 评估模型
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
#
# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("Mean Absolute Error:", mae)
# print("R² Score:", r2)
# print("MAPE:", mape)
#
# # 计算 Adjusted R²
# n = len(y_test)  # 样本数
# p = X_train.shape[1]  # 自变量数量
# adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
# print("Adjusted R² Score:", adjusted_r2)
#
# # 6. 特征重要性
# importances = best_rf.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# # 输出特征重要性
# print("Feature ranking:")
# for f in range(X_preprocessed.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
# # 7. 交叉验证
# scores = cross_val_score(best_rf, X_preprocessed, y, cv=5, scoring='neg_mean_squared_error')  # 使用负的MSE进行交叉验证
# print("交叉验证均方误差：", -np.mean(scores))  # 输出平均的MSE
#
# # 8. 绘制学习曲线
# train_sizes, train_scores, test_scores = learning_curve(best_rf, X_preprocessed, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='neg_mean_squared_error')
# train_mean = -np.mean(train_scores, axis=1)  # 计算训练得分的平均值
# train_std = np.std(train_scores, axis=1)  # 计算训练得分的标准差
# test_mean = -np.mean(test_scores, axis=1)  # 计算交叉验证得分的平均值
# test_std = np.std(test_scores, axis=1)  # 计算交叉验证得分的标准差
#
# plt.figure(figsize=(12, 8))  # 创建绘图窗口
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")  # 训练得分标准差区间
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")  # 交叉验证得分标准差区间
# plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")  # 绘制训练得分曲线
# plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")  # 绘制交叉验证得分曲线
# plt.xlabel("Training examples")  # x轴标签
# plt.ylabel("Score (MSE)")  # y轴标签
# plt.legend(loc="best")  # 添加图例
# plt.savefig('rf_kdd.png')  # 保存绘制的图形
# plt.show()
#
# # 9. 使用 SHAP 解释模型
# explainer = shap.TreeExplainer(best_rf)  # 使用最佳模型
# shap_values = explainer.shap_values(X_test)
#
# # 可视化特征重要性
# shap.summary_plot(shap_values, X_test)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import json
import seaborn as sns

# 1. 加载 JSONL 文件
file_path = '/mnt/random_forest/task1/new_data1.jsonl'  # 修改为您的 JSONL 文件路径

# 读取 JSONL 文件
data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 显示数据的前几行，以确认数据是否正确加载
print("原始数据：")
print(df.head().to_string(index=False))

# 2. 数据预处理
# 删除第一列（日期），请根据需要调整
df.drop(df.columns[0], axis=1, inplace=True)

# 检查是否有缺失值
if df.isnull().values.any():
    print("警告：存在缺失值")

# 将数据分为特征和目标变量
X = df.iloc[:, :-1]  # 所有的特征
y = df.iloc[:, -1]   # 目标变量（野生羊肚菌干产量）

# 创建一个简单的填充器和标准化处理器
imputer = SimpleImputer(strategy='mean')  # 使用平均值填充缺失值
scaler = StandardScaler()  # 标准化处理器

# 预处理步骤
preprocessor = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler),
])

# 对特征数据应用预处理器
X_preprocessed = preprocessor.fit_transform(X)

# 输出归一化后的数据前几行
print("归一化后的数据：")
print(pd.DataFrame(X_preprocessed).head())

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# 4. 创建随机森林回归模型
rf = RandomForestRegressor(random_state=42)

# 超参数调优
param_grid = {
    'n_estimators': [100, 200, 300],  # 决策树数量
    'max_depth': [10, 20, 30],  # 每棵树的最大深度
    'min_samples_split': [5, 10, 15],  # 内部节点再次分裂所需最小样本数
    'min_samples_leaf': [1, 2, 4]  # 叶子节点所需最小样本数
}

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 5. 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
print("MAPE:", mape)

# 计算 Adjusted R²
n = len(y_test)  # 样本数
p = X_train.shape[1]  # 自变量数量
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R² Score:", adjusted_r2)

# 6. 特征重要性
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 输出特征重要性
print("Feature ranking:")
for f in range(X_preprocessed.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# 7. 交叉验证
scores = cross_val_score(best_rf, X_preprocessed, y, cv=5, scoring='neg_mean_squared_error')  # 使用负的MSE进行交叉验证
print("交叉验证均方误差：", -np.mean(scores))  # 输出平均的MSE

# 8. 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(best_rf, X_preprocessed, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='neg_mean_squared_error')
train_mean = -np.mean(train_scores, axis=1)  # 计算训练得分的平均值
train_std = np.std(train_scores, axis=1)  # 计算训练得分的标准差
test_mean = -np.mean(test_scores, axis=1)  # 计算交叉验证得分的平均值
test_std = np.std(test_scores, axis=1)  # 计算交叉验证得分的标准差

plt.figure(figsize=(12, 8))  # 创建绘图窗口
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")  # 训练得分标准差区间
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")  # 交叉验证得分标准差区间
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")  # 绘制训练得分曲线
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")  # 绘制交叉验证得分曲线
plt.xlabel("Training examples")  # x轴标签
plt.ylabel("Score (MSE)")  # y轴标签
plt.legend(loc="best")  # 添加图例
plt.savefig('rf_kdd.png')  # 保存绘制的图形
plt.show()

# 9. 使用 SHAP 解释模型
explainer = shap.TreeExplainer(best_rf)  # 使用最佳模型
shap_values = explainer.shap_values(X_test)

# 可视化特征重要性
shap.summary_plot(shap_values, X_test)

# 10. 绘制特征重要性热图
feature_importance_df = pd.DataFrame({
    'Feature': df.columns[:-1],
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.heatmap(feature_importance_df.set_index('Feature').T, cmap='viridis', annot=True, cbar=True)
plt.title("Feature Importance Heatmap")
plt.savefig('feature_importance_heatmap.png')  # 保存热图
plt.show()

