import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from math import sqrt

# 读取 JSONL 文件数据
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# 数据预处理，提取特征和目标变量，并划分训练集和测试集
def preprocess_data(data, test_size=0.2):
    features = []
    targets = []
    for entry in data:
        features.append([entry[key] for key in entry if key != "野生羊肚菌干产量/Kg"])
        targets.append(entry["野生羊肚菌干产量/Kg"])

    # 转换为 NumPy 数组
    features = np.array(features)
    targets = np.array(targets)

    # 归一化特征
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=42)

    # 转换为 PyTorch 的张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, X_test, y_train, y_test, scaler


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 模型训练
def train_model(model, features, targets, batch_size, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(features, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs.unsqueeze(1))  # 添加时间维度
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')


# 模型评估
def evaluate_model(model, features, targets):
    model.eval()
    with torch.no_grad():
        predictions = model(features.unsqueeze(1))
        mse = mean_squared_error(targets, predictions)
        rmse = sqrt(mse)
        mape = mean_absolute_percentage_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2 Score: {r2:.4f}')
        return mse, rmse, mape, r2


# 贝叶斯优化的目标函数
def objective(trial):
    # 定义要调节的超参数空间
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 5000, 10000)

    # 定义模型
    model = LSTMModel(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1)

    # 训练模型
    train_model(model, X_train, y_train, batch_size, num_epochs, learning_rate)

    # 评估模型在测试集上的表现
    mse, rmse, mape, r2 = evaluate_model(model, X_test, y_test)

    # 返回优化目标值（例如 MSE）
    return mse


# 主函数
def main():
    # 加载并预处理数据
    file_path = '/mnt/random_forest/data/output_dataset.jsonl'  # 替换为你的文件路径
    data = load_jsonl(file_path)
    global X_train, X_test, y_train, y_test, scaler
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # 使用 optuna 进行贝叶斯优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # 输出最佳超参数
    print("Best hyperparameters: ", study.best_params)

    # 使用最优超参数训练最终模型
    best_params = study.best_params
    model = LSTMModel(input_size=X_train.shape[1], hidden_size=best_params['hidden_size'],
                      num_layers=best_params['num_layers'], output_size=1)

    train_model(model, X_train, y_train, batch_size=best_params['batch_size'],
                num_epochs=best_params['num_epochs'], learning_rate=best_params['learning_rate'])

    # 最终模型在训练集上的评估
    print("Final Training Set Evaluation:")
    evaluate_model(model, X_train, y_train)

    # 最终模型在测试集上的评估
    print("Final Test Set Evaluation:")
    evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()
