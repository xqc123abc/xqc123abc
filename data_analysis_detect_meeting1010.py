import re
import os
import json
import pandas as pd
import numpy as np
from io import StringIO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def get_charge_discharge_rates(target_simulation_num,log_file_path='JCJQ_Normal_data/simulation_log.txt'):
    """
    从模拟日志文件中提取指定编号对应的充电倍率和放电倍率
    
    参数：
    log_file_path: str - 模拟日志文件（simulation_log.txt）的路径
    target_simulation_num: int - 目标模拟编号（如 10，对应 Simulation 10）
    
    返回：
    tuple - (充电倍率 rate_ch, 放电倍率 rate_dch)，若未找到则返回 (None, None)
    """
    # 定义匹配日志行的正则表达式：提取 "Simulation X" 和后面的字典字符串
    # 例如匹配 "[2025-09-16 00:34:49] Simulation 1: {'E_max': 36.872, ...}"
    pattern = re.compile(
        r"Simulation (\d+): ({.*?})",  
        re.DOTALL  # 确保匹配跨换行的内容（若日志有换行）
    )

    try:
        # 读取日志文件内容
        with open(log_file_path, 'r', encoding='gbk') as f:
            log_content = f.read()

        # 查找所有符合格式的模拟记录
        all_simulations = pattern.findall(log_content)

        # 遍历所有记录，匹配目标编号
        for sim_num_str, param_str in all_simulations:
            sim_num = int(sim_num_str)
            if sim_num == target_simulation_num:
                # 将字符串格式的字典转换为 Python 字典（处理单引号为双引号，符合JSON格式）
                param_dict = json.loads(param_str.replace("'", '"'))
                # 提取充电和放电倍率
                rate_ch = param_dict.get('rate_ch')
                rate_dch = param_dict.get('rate_dch')
                return rate_ch, rate_dch

        # 若遍历完未找到目标编号
        print(f"错误：未在日志文件中找到编号为 {target_simulation_num} 的模拟记录")
        return None, None

    except FileNotFoundError:
        print(f"错误：未找到文件 {log_file_path}，请检查文件路径是否正确")
        return None, None
    except json.JSONDecodeError:
        print(f"错误：编号 {target_simulation_num} 的参数格式异常，无法解析")
        return None, None
    except Exception as e:
        print(f"程序运行出错：{str(e)}")
        return None, None


def extract_time_ucell(filepath='U_100.txt') -> pd.DataFrame:
    """
    从 COMSOL 导出的 CSV 文件中提取“时间”和“Ucell_1-1”列。
    
    参数：
        filepath (str): CSV 文件路径
        
    返回：
        pd.DataFrame: 仅包含“时间”和“Ucell_1-1”两列的数据
    """
    # 过滤掉以 % 开头的注释行
    filepath = os.path.join('JCJQ_Normal_data', filepath)
    with open(filepath, 'r', encoding='gbk') as f:
        lines = [line.strip() for line in f if not line.startswith('%') and line.strip() != '']

    # 构建临时 CSV 内容并读取
    text = "\n".join(lines)
    text = text.replace(';', ',')
    df = pd.read_csv(StringIO(text),header=None)
    df.set_index(df.columns[0], inplace=True)

    # 提取目标列
    # !! 注意，此处只提取一个单体的前25个时刻
    return df.iloc[-25:,:1]


def load_voltage_tem_tensors(folder='JCJQ_Normal_data'):
    """
    批量读取电压和温度文件，并转换为单个张量，进行全局归一化。
    
    返回：
        volt_tensor: Tensor，[num_samples, time_steps]，电压归一化到 [0,1]
        tem_tensor: Tensor，[num_samples, time_steps]，温度归一化到 [0,1]
        current_tensor: Tensor，[num_samples, 2]，每个样本的充放电倍率 [rate_ch, rate_dch]
        time_list: list[pd.Index]，每个样本的时间索引
    """
    voltage_files = sorted([f for f in os.listdir(folder) if f.startswith('U_')])
    tem_files = sorted([f for f in os.listdir(folder) if f.startswith('T_')])

    n_pairs = min(len(voltage_files), len(tem_files))
    print(f"检测到 {n_pairs} 对电压-温度文件。")

    all_v_values, all_t_values, current_list, time_list = [], [], [], []

    # 读取所有样本数据
    for i in range(0,n_pairs,30):
        # 注意，仅仅取了部分文件
        df_v = extract_time_ucell(voltage_files[i])
        df_t = extract_time_ucell(tem_files[i])

        common_time = df_v.index.intersection(df_t.index)
        v_values = df_v.loc[common_time].iloc[:, 0].to_numpy(dtype='float32')
        t_values = df_t.loc[common_time].iloc[:, 0].to_numpy(dtype='float32')

        all_v_values.append(v_values)
        all_t_values.append(t_values)
        time_list.append(common_time)

        # 充放电倍率
        rate_ch, rate_dch = get_charge_discharge_rates(i+1)
        current_list.append([rate_ch, rate_dch])

    # 将列表转换为 NumPy 数组方便计算全局归一化
    all_v_array = np.stack(all_v_values)  # shape: [num_samples, time_steps]
    all_t_array = np.stack(all_t_values)

    # 全局归一化
    v_min, v_max = all_v_array.min(), all_v_array.max()
    t_min, t_max = all_t_array.min(), all_t_array.max()

    print(f"电压全局归一化: min={v_min}, max={v_max}")
    print(f"温度全局归一化: min={t_min}, max={t_max}")

    all_v_array = (all_v_array - v_min) / (v_max - v_min)
    all_t_array = (all_t_array - t_min) / (t_max - t_min)

    # 转换为张量
    volt_tensor = torch.tensor(all_v_array, dtype=torch.float32)
    tem_tensor = torch.tensor(all_t_array, dtype=torch.float32)
    current_tensor = torch.tensor(current_list, dtype=torch.float32)

    return volt_tensor, tem_tensor, current_tensor, time_list


# 构建 PyTorch Dataset
class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, input_seq, rate_ch, rate_dch, target):
            self.input_seq = input_seq
            self.rate_ch = rate_ch
            self.rate_dch = rate_dch
            self.target = target

        def __len__(self):
            return self.input_seq.shape[0]

        def __getitem__(self, idx):
            return (self.input_seq[idx], self.rate_ch[idx], self.rate_dch[idx], self.target[idx])

class PredictModel(nn.Module):
        def __init__(self, time_steps=5, embedding_dim=4, hidden_dim=16, num_layers=1):
            super().__init__()
            # 假设充放电倍率离散值最大为 11（根据实际修改）
            self.rate_ch_embedding = nn.Embedding(11, embedding_dim)
            self.rate_dch_embedding = nn.Embedding(11, embedding_dim)

            self.lstm = nn.LSTM(input_size=2 + 2*embedding_dim, hidden_size=hidden_dim,
                                num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 2)  # 输出温度和电压预测
            self.time_fc = nn.Linear(20, time_steps)  # 输出温度和电压预测

        def forward(self, x_seq, rate_ch, rate_dch):
            # x_seq: [batch, time_steps, 2]
            ch_emb = self.rate_ch_embedding(rate_ch)  # [batch, embedding_dim]
            dch_emb = self.rate_dch_embedding(rate_dch)  # [batch, embedding_dim]
            # 扩展到时间步长维度
            ch_emb = ch_emb.unsqueeze(1).repeat(1, x_seq.shape[1], 1)
            dch_emb = dch_emb.unsqueeze(1).repeat(1, x_seq.shape[1], 1)

            x = torch.cat([x_seq, ch_emb, dch_emb], dim=2)  # [batch, time_steps, 2+embedding*2]
            out, _ = self.lstm(x)
            # out = out[:, [-1], :]  # 取最后1个时间步的输出
            out = out[:, -5:, :]  # 取最后五个时间步的输出
            out = self.fc(out)
            # out = self.time_fc(torch.permute(out,(0,2,1))).permute(0,2,1)
            return out     


def plot_multistep_prediction(model, dataset, pred_length=5,sample_idx=0, device='cpu'):
    """
    可视化温度、电压多步预测结果（预测未来5个时刻）
    
    参数：
        model: 训练好的 PyTorch 模型
        dataset: PyTorch Dataset，包含 (x_seq, rate_ch, rate_dch, target)
                 target 应为未来5个时刻的真实值 [5, 2]
        sample_idx: int，要绘制的样本索引
        device: str 或 torch.device，模型和数据使用的设备
    """
    model.eval()
    
    # 获取样本
    x_seq, ch, dch, y_true = dataset[sample_idx]
    x_seq = x_seq.unsqueeze(0).to(device)
    ch = ch.unsqueeze(0).to(device)
    dch = dch.unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_pred = model(x_seq, ch, dch).cpu().numpy()[0]  # shape [5,2] 未来5个时刻
    y_true = y_true.numpy()  # shape [5,2]

    # 前20个时刻真实值
    tem_20 = x_seq[0, :, 0].cpu().numpy()
    volt_20 = x_seq[0, :, 1].cpu().numpy()
    
    time_steps = range(1, 21)
    # future_steps = range(20, 26)  # 未来5个时刻
    if pred_length == 1:
        future_steps = (20,25)
    else:
        future_steps = range(20, 21+pred_length)

    plt.figure(figsize=(10,5))
    
    # 温度绘制
    plt.plot(time_steps, tem_20, label='Temperature (t1-t20)', color='blue', marker='o')
    plt.plot(future_steps, np.insert(y_true[:,0],0, tem_20[-1]), label='True Temp t21-t25', color='navy', marker='x')
    plt.plot(future_steps, np.insert(y_pred[:,0],0, tem_20[-1]), label='Predicted Temp t21-t25', color='cyan', marker='^')
    
    # 电压绘制
    plt.plot(time_steps, volt_20, label='Voltage (t1-t20)', color='orange', marker='o')
    plt.plot(future_steps, np.insert(y_true[:,1],0, volt_20[-1]), label='True Volt t21-t25', color='red', marker='x')
    plt.plot(future_steps, np.insert(y_pred[:,1],0, volt_20[-1]), label='Predicted Volt t21-t25', color='magenta', marker='^')
    
    # 在 t20 与 t21 之间画红色竖线
    plt.axvline(x=20.5, color='red', linestyle='--', linewidth=2)

    plt.xlabel('Time step')
    plt.ylabel('Normalized value')
    plt.title(f'Sample {sample_idx} - Temperature and Voltage Multi-step Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    volt_tensors, tem_tensors, current_list, time_list = load_voltage_tem_tensors()

    

    # ---------------------------
    # 1. 数据准备
    # ---------------------------
    # 假设 time_steps = 20
    num_samples, time_steps = volt_tensors.shape

    # 离散化充放电倍率（假设原本就是整数）
    # 可以使用 embedding 编码
    rate_ch = (10*current_list[:, 0]).long()
    rate_dch = (10*current_list[:, 1]).long()

    # 构建输入特征
    # 输入特征为 [温度, 电压] + embedding(rate_ch) + embedding(rate_dch)
    # 首先把温度和电压扩展一个维度，方便和 embedding 拼接
    input_features = torch.stack([tem_tensors[:, :-5], volt_tensors[:, :-5]], dim=2)  # shape: [num_samples, time_steps, 2]

    # ---------------------------
    # 2. 数据集划分
    # ---------------------------
    dataset_size = num_samples
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    # 目标为下一个时刻第1个值
    # target = torch.stack([tem_tensors[:, [-1]], volt_tensors[:, [-1]]], dim=2)  # [num_samples, 2]
    target = torch.stack([tem_tensors[:, -5:], volt_tensors[:, -5:]], dim=2)  # [num_samples, 2]

    dataset = TimeSeriesDataset(input_features, rate_ch, rate_dch, target)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ---------------------------
    # 3. 模型定义
    # ---------------------------
    model = PredictModel().to(device)

    # ---------------------------
    # 4. 训练
    # ---------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 500

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_seq, ch, dch, y in train_loader:
            x_seq, ch, dch, y = x_seq.to(device), ch.to(device), dch.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x_seq, ch, dch)
            print(y_pred.shape, y.shape)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_seq.size(0)

        train_loss /= train_size

        # 测试集评估
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_seq, ch, dch, y in test_loader:
                x_seq, ch, dch, y = x_seq.to(device), ch.to(device), dch.to(device), y.to(device)
                y_pred = model(x_seq, ch, dch)
                loss = criterion(y_pred, y)
                test_loss += loss.item() * x_seq.size(0)
        test_loss /= test_size

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Test Loss: {test_loss:.6f}")


    # ---------------------------
    # 5. 保存模型
    # ---------------------------
    model_path = 'Models/detect/new_data_model.pt'
    os.makedirs('Models/detect', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    # ---------------------------
    # 6. 可视化
    # ---------------------------
    plot_multistep_prediction(model, test_dataset, sample_idx=10, device=device)
