import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


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
    feature_scaler = MinMaxScaler()
    features = feature_scaler.fit_transform(features)

    # 归一化目标变量
    target_scaler = MinMaxScaler()
    targets = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=42)

    # 转换为 PyTorch 的张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler


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


# 模型训练，添加早停机制
def train_model(model, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.01, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train.unsqueeze(1))  # 添加时间维度
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证集上的损失
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.unsqueeze(1))
            val_loss = criterion(val_outputs, y_val)

        # 检查是否早停
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch + 1}")
            early_stop = True
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        if early_stop:
            break


# 模型评估
def evaluate_model(model, features, targets, target_scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(features.unsqueeze(1))

        # 反标准化目标变量和预测结果
        predictions_original = target_scaler.inverse_transform(predictions.detach().cpu().numpy())
        targets_original = target_scaler.inverse_transform(targets.detach().cpu().numpy())

        # 计算评估指标
        mse = mean_squared_error(targets_original, predictions_original)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(targets_original, predictions_original)
        r2 = r2_score(targets_original, predictions_original)

        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAPE: {mape:.4f}')
        print(f'R² Score: {r2:.4f}')

    return predictions_original


# 主函数
def main():
    # 加载并预处理数据
    file_path = '/mnt/random_forest/data/output_dataset.jsonl'  # 替换为你的文件路径
    data = load_jsonl(file_path)
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler = preprocess_data(data)

    # 定义模型参数
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 2
    output_size = 1

    # 初始化模型
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 将训练集划分为训练集和验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 训练模型，使用早停
    train_model(model, X_train_split, y_train_split, X_val_split, y_val_split, num_epochs=200, patience=10)

    # 评估模型在训练集上的表现
    print("Training Set Evaluation:")
    _ = evaluate_model(model, X_train, y_train, target_scaler)

    # 评估模型在测试集上的表现
    print("Test Set Evaluation:")
    predictions = evaluate_model(model, X_test, y_test, target_scaler)


if __name__ == '__main__':
    main()
