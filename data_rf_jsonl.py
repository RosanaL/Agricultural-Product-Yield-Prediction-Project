import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import json

def train_model(df):
    X = df.drop(columns=['野生羊肚菌干产量/Kg'])
    y = df['野生羊肚菌干产量/Kg']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return model, scaler


def predict_and_replace(excel_file, output_file):
    df = pd.read_excel(excel_file)
    df_non_zero = df[df['野生羊肚菌干产量/Kg'] != 0]

    model, scaler = train_model(df_non_zero)

    df_zero = df[df['野生羊肚菌干产量/Kg'] == 0]
    if not df_zero.empty:
        X_zero_scaled = scaler.transform(df_zero.drop(columns=['野生羊肚菌干产量/Kg']))
        df_zero.loc[:, '野生羊肚菌干产量/Kg'] = model.predict(X_zero_scaled).round(1)

    df_updated = pd.concat([df_non_zero, df_zero])

    # 输出为JSONL文件，确保使用UTF-8编码
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in df_updated.to_dict(orient='records'):
            f.write(f"{json.dumps(record, ensure_ascii=False)}\n")  # 使用json.dumps确保正确格式


# 使用函数，替换为你的Excel文件路径和输出文件路径
excel_file = '/mnt/random_forest/task1/DATA SET.xlsx'
output_file = '/mnt/random_forest/task1/data1.jsonl'
predict_and_replace(excel_file, output_file)
