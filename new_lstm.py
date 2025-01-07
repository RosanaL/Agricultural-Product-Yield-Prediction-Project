import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
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

# 定义增强版 LSTM 模型（去掉 Dropout）
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取 LSTM 的最后一个时间步的输出
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

# 模型训练
def train_model(model, features, targets, num_epochs=100, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(features.unsqueeze(1))  # 添加时间维度
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型评估
def evaluate_model(model, features, targets):
    model.eval()
    with torch.no_grad():
        predictions = model(features.unsqueeze(1))
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        print(f'MSE: {mse:.4f}, R2 Score: {r2:.4f}')
        return predictions

# 主函数
def main():
    # 加载并预处理数据
    file_path = '/mnt/random_forest/data/output_dataset.jsonl'  # 替换为你的文件路径
    data = load_jsonl(file_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # 定义模型参数
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 3  # 增加 LSTM 层数
    output_size = 1

    # 初始化模型
    model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size)

    # 训练模型
    train_model(model, X_train, y_train, num_epochs=15000)

    # 评估模型在训练集上的表现
    print("Training Set Evaluation:")
    _ = evaluate_model(model, X_train, y_train)

    # 评估模型在测试集上的表现
    print("Test Set Evaluation:")
    predictions = evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
