# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def train_model(df):
    # 提取特征和目标变量
    X = df.drop(columns=['野生羊肚菌干产量/Kg'])
    y = df['野生羊肚菌干产量/Kg']

    # 归一化处理特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 创建并训练随机森林回归模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测测试集的结果
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R? Score: {r2:.2f}")

    return model

def predict_and_replace(excel_file, output_file):
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 过滤掉产量为0的数据，用于训练模型
    df_non_zero = df[df['野生羊肚菌干产量/Kg'] != 0]

    # 归一化处理特征
    scaler = MinMaxScaler()
    X_non_zero_scaled = scaler.fit_transform(df_non_zero.drop(columns=['野生羊肚菌干产量/Kg']))

    # 训练模型
    model = train_model(df_non_zero)

    # 找到产量为0的行并进行预测
    df_zero = df[df['野生羊肚菌干产量/Kg'] == 0]
    if not df_zero.empty:
        X_zero_scaled = scaler.transform(df_zero.drop(columns=['野生羊肚菌干产量/Kg']))
        df_zero.loc[:, '野生羊肚菌干产量/Kg'] = model.predict(X_zero_scaled).round(1)  # 使用.loc 和 round(1)

    # 合并已修改的部分与未修改的部分
    df_updated = pd.concat([df_non_zero, df_zero])

    # 将结果写入新的Excel文件
    df_updated.to_excel(output_file, index=False)

# 使用函数，替换为你的Excel文件路径和输出文件路径
excel_file = '/mnt/random_forest/task1/DATA SET.xlsx'
output_file = '/mnt/random_forest/task1/data1.xlsx'
predict_and_replace(excel_file, output_file)
