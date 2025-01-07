import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def train_model(df):
    # ��ȡ������Ŀ�����
    X = df.drop(columns=['Ұ����Ǿ��ɲ���/Kg'])
    y = df['Ұ����Ǿ��ɲ���/Kg']

    # ��һ����������
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # �����ݷ�Ϊѵ�����Ͳ��Լ�
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ������ѵ�����ɭ�ֻع�ģ��
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Ԥ����Լ��Ľ��
    y_pred = model.predict(X_test)

    # ��������ָ��
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R? Score: {r2:.2f}")

    return model

def predict_and_replace(excel_file, output_file):
    # ��ȡExcel�ļ�
    df = pd.read_excel(excel_file)

    # ���˵�����Ϊ0�����ݣ�����ѵ��ģ��
    df_non_zero = df[df['Ұ����Ǿ��ɲ���/Kg'] != 0]

    # ��һ����������
    scaler = MinMaxScaler()
    X_non_zero_scaled = scaler.fit_transform(df_non_zero.drop(columns=['Ұ����Ǿ��ɲ���/Kg']))

    # ѵ��ģ��
    model = train_model(df_non_zero)

    # �ҵ�����Ϊ0���в�����Ԥ��
    df_zero = df[df['Ұ����Ǿ��ɲ���/Kg'] == 0]
    if not df_zero.empty:
        X_zero_scaled = scaler.transform(df_zero.drop(columns=['Ұ����Ǿ��ɲ���/Kg']))
        df_zero.loc[:, 'Ұ����Ǿ��ɲ���/Kg'] = model.predict(X_zero_scaled).round(1)  # ʹ��.loc �� round(1)

    # �ϲ����޸ĵĲ�����δ�޸ĵĲ���
    df_updated = pd.concat([df_non_zero, df_zero])

    # �����д���µ�Excel�ļ�
    df_updated.to_excel(output_file, index=False)

# ʹ�ú������滻Ϊ���Excel�ļ�·��������ļ�·��
excel_file = '/mnt/random_forest/data/dataset.xlsx'
output_file = '/mnt/random_forest/data/output_dataset.xlsx'
predict_and_replace(excel_file, output_file)
