import argparse
import json
import struct
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import socket
import sys
import io
import threading
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
import time
import can
import re
from io import StringIO
# from myCAN import CANReceiver
# from CANHub import can_hub
# import multiprocessing

print('torch version:', torch.__version__)
# plt.rcParams['font.family'] = 'SimHei'
# 设置标准输出的编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 在程序中设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# fault_name_eng = ['SOH-low', 'cellshortcircuit', 'voltage-diff-high', 'cellopencircuit', 'pressure-high', 'lowcapacity', \
#                   'packshortcircuit', 'packopencircuit', 'thermorunaway', 'electrolyteleakage']
# fault_name_ch = ['单体SOH偏低', '单体短路', '单体间电压差过大', '单体开路', '单体气压过高', '单体容量偏低', '电池组短路', '电池组开路', '温度过高', '泄露']
# class_name_ch = ['正常', '单体SOH偏低', '单体短路',  '单体间电压差过大', '单体开路', '单体气压过高', '单体容量偏低', '电池组短路', '电池组开路', '温度过高', '泄露']
fault_name_eng = ['cellopencircuit', 'cellshortcircuit', 'electrolyteleakage', 'lowcapacity', 'packopencircuit', 'packshortcircuit',
                  'pressure-high', 'SOH-low', 'thermorunaway', 'voltage-diff-high']
fault_name_ch = ['单体开路', '单体短路', '泄漏', '单体容量偏低', '电池组开路', '电池组短路', '单体气压过高', '单体SOH偏低', '温度过高', '单体间电压差过大']
class_name_ch = ['正常', '单体开路', '单体短路', '泄漏', '单体容量偏低', '电池组开路', '电池组短路', '单体气压过高', '单体SOH偏低', '温度过高', '单体间电压差过大']
# features = ['蓄电池组A电压遥测','蓄电池组A放电电流遥测', '蓄电池组A充电电流遥测','蓄电池组A温度1', '蓄电池组A温度2', '蓄电池组A温度3',
#                         '蓄电池组B电压遥测','蓄电池组B放电电流遥测', '蓄电池组B充电电流遥测','蓄电池组B温度1', '蓄电池组B温度2', '蓄电池组B温度3',
#                         '锂电池A单体1电压', '锂电池B单体1电压', '锂电池A单体2电压', '锂电池B单体2电压', '锂电池A单体3电压', '锂电池B单体3电压', '锂电池A单体4电压',
#                          '锂电池B单体4电压', '锂电池A单体5电压', '锂电池B单体5电压', '锂电池A单体6电压', '锂电池B单体6电压', '锂电池A单体7电压', '锂电池B单体7电压']
packFeatures_voltage = ['蓄电池组A电压遥测', '蓄电池组B电压遥测']
packFeatures_current = ['蓄电池组A放电电流遥测', '蓄电池组A充电电流遥测', '蓄电池组B放电电流遥测', '蓄电池组B充电电流遥测']
packFeatures_temperature1 = ['蓄电池组A温度1','蓄电池组B温度1']
packFeatures_temperature2 = ['蓄电池组A温度2','蓄电池组B温度2']
packFeatures_temperature3 = ['蓄电池组A温度3','蓄电池组B温度3']
features_voltageA = ['锂电池A单体1电压', '锂电池A单体2电压', '锂电池A单体3电压', '锂电池A单体4电压', '锂电池A单体5电压', '锂电池A单体6电压', '锂电池A单体7电压']
features_voltageB = ['锂电池B单体1电压', '锂电池B单体2电压', '锂电池B单体3电压', '锂电池B单体4电压', '锂电池B单体5电压', '锂电池B单体6电压', '锂电池B单体7电压']
features = ['蓄电池组A电压遥测','蓄电池组A放电电流遥测', '蓄电池组A充电电流遥测','蓄电池组A温度1', '蓄电池组A温度2', '蓄电池组A温度3',
                        '蓄电池组B电压遥测','蓄电池组B放电电流遥测', '蓄电池组B充电电流遥测','蓄电池组B温度1', '蓄电池组B温度2', '蓄电池组B温度3',
                        '锂电池A单体1电压', '锂电池A单体2电压', '锂电池A单体3电压', '锂电池A单体4电压', '锂电池A单体5电压', '锂电池A单体6电压', '锂电池A单体7电压',
                         '锂电池B单体1电压', '锂电池B单体2电压', '锂电池B单体3电压', '锂电池B单体4电压', '锂电池B单体5电压', '锂电池B单体6电压', '锂电池B单体7电压']

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


def detect_data_type(data_path):
    """
    自动判断数据类型：老数据还是新数据
    
    参数：
        data_path: str - 数据路径
    
    返回：
        str - 'old_data' 或 'new_data'
    """
    if os.path.isfile(data_path):
        # 单个文件，检查是否为老数据格式
        try:
            with open(data_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip()
                if '系统时间' in first_line or '蓄电池组' in first_line:
                    return 'old_data'
        except:
            pass
    elif os.path.isdir(data_path):
        # 目录，检查是否包含新数据格式文件
        files = os.listdir(data_path)
        if any(f.startswith('U_') and f.endswith('.csv') for f in files):
            return 'new_data'
    
    # 默认返回老数据
    return 'old_data'


def get_charge_discharge_rates(target_simulation_num, log_file_path='JCJQ_Normal_data/simulation_log.txt'):
    """
    从模拟日志文件中提取指定编号对应的充电倍率和放电倍率
    
    参数：
    log_file_path: str - 模拟日志文件（simulation_log.txt）的路径
    target_simulation_num: int - 目标模拟编号（如 10，对应 Simulation 10）
    
    返回：
    tuple - (充电倍率 rate_ch, 放电倍率 rate_dch)，若未找到则返回 (None, None)
    """
    # 定义匹配日志行的正则表达式：提取 "Simulation X" 和后面的字典字符串
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


def extract_time_ucell(filepath='U_100.txt'):
    """
    从 COMSOL 导出的 CSV 文件中提取"时间"和"Ucell_1-1"列。
    
    参数：
        filepath (str): CSV 文件路径
        
    返回：
        pd.DataFrame: 仅包含"时间"和"Ucell_1-1"两列的数据
    """
    # 过滤掉以 % 开头的注释行
    filepath = os.path.join('JCJQ_Normal_data', filepath)
    with open(filepath, 'r', encoding='gbk') as f:
        lines = [line.strip() for line in f if not line.startswith('%') and line.strip() != '']

    # 构建临时 CSV 内容并读取
    text = "\n".join(lines)
    text = text.replace(';', ',')
    df = pd.read_csv(StringIO(text), header=None)
    df.set_index(df.columns[0], inplace=True)

    # 提取目标列
    # !! 注意，此处只提取一个单体的前25个时刻
    return df.iloc[-25:, :1]


def load_new_data_tensors(folder='JCJQ_Normal_data'):
    """
    批量读取新数据（电压和温度文件），并转换为张量，进行全局归一化。
    
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
    for i in range(0, n_pairs, 30):  # 每30个文件取一个样本
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


class NewDataPredictModel(nn.Module):
    """新数据的预测模型"""
    def __init__(self, time_steps=5, embedding_dim=4, hidden_dim=16, num_layers=1):
        super().__init__()
        # 假设充放电倍率离散值最大为 11（根据实际修改）
        self.rate_ch_embedding = nn.Embedding(11, embedding_dim)
        self.rate_dch_embedding = nn.Embedding(11, embedding_dim)

        self.lstm = nn.LSTM(input_size=2 + 2*embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # 输出温度和电压预测

    def forward(self, x_seq, rate_ch, rate_dch):
        # x_seq: [batch, time_steps, 2]
        ch_emb = self.rate_ch_embedding(rate_ch)  # [batch, embedding_dim]
        dch_emb = self.rate_dch_embedding(rate_dch)  # [batch, embedding_dim]
        # 扩展到时间步长维度
        ch_emb = ch_emb.unsqueeze(1).repeat(1, x_seq.shape[1], 1)
        dch_emb = dch_emb.unsqueeze(1).repeat(1, x_seq.shape[1], 1)

        x = torch.cat([x_seq, ch_emb, dch_emb], dim=2)  # [batch, time_steps, 2+embedding*2]
        out, _ = self.lstm(x)
        out = out[:, -5:, :]  # 取最后五个时间步的输出
        out = self.fc(out)
        return out


def load_new_data_model(model_path='Models/detect/new_data_model.pt', device='cpu'):
    """加载新数据的训练好的模型"""
    try:
        model = NewDataPredictModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"成功加载模型: {model_path}")
        return model
    except FileNotFoundError:
        print(f"警告：未找到新数据模型文件 {model_path}，将使用随机初始化的模型")
        print("请先运行 data_analysis_detect_meeting1010.py 训练并保存模型")
        return NewDataPredictModel().to(device)
    except Exception as e:
        print(f"加载新数据模型时出错：{e}")
        print("请确保模型文件格式正确")
        return NewDataPredictModel().to(device)


def detect_null(data):
    # print('-' * 10, '开始检测nan值', '-' * 10)
    # 输出缺失值null坐标
    nan_positions = data.isnull().values
    nan_positions = data[nan_positions].index.to_list()
    if len(nan_positions) > 0:
        print(nan_positions)
    # print('-' * 10, '结束', '-' * 10)
    return nan_positions


def get_3sigma_abnormal(data):
    # print('-' * 10, '开始检测异常值', '-' * 10)
    # 清理异常值
    data_normal = data.copy()
    means = data_normal.mean()
    stds = data_normal.std()
    normal_index = ((data_normal - means).abs() <= 3 * stds).all(axis=1).values

    # 反转逻辑：取异常值所在的行索引
    abnormal_indices = data.index[~normal_index]

    if len(abnormal_indices) > 0:
        print('异常值坐标：')
        print(abnormal_indices.tolist())  # 转为列表可读性更高
    else:
        print('没有检测到异常值。')

    # print('-' * 10, '结束', '-' * 10)

    return data.iloc[normal_index, :].copy(), data.iloc[~normal_index, :].copy()

def get_labels(data, fault2cell):
    # 给每个故障类型一个唯一编号
    fault_map = {fault: idx + 1 for idx, fault in enumerate(fault2cell.keys())}

    # 读取数据

    # 初始化labels
    labels = pd.DataFrame(0, index=data.index, columns=range(1, 15))

    # 遍历每行
    for i, tag in enumerate(data['tag']):
        if tag in fault2cell:
            cols = fault2cell[tag]  # 单体编号（1~14）
            code = fault_map[tag]  # 对应编号
            labels.loc[i, cols] = code  # 给对应列赋值

    # 如果需要 numpy 数组
    labels_array = labels.to_numpy()
    return labels_array

# 归一化
def get_scaler(data):
    # id to predict: cell id for detect
    #   A: 1 3 5 7
    #   B: 2 4 6 8

    # print('-' * 10, '开始归一化', '-' * 10)
    # 数据归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler = StandardScaler(with_std=False)
    data_stander = data.copy()
    data_stander = data_stander.iloc[:, (data_stander.std(axis=0) > 1e-5).values]
    # if not os.path.exists(os.path.join(saved_path, f'cell{"".join([str(i) for i in id_to_predict])}')):
    #     os.makedirs(os.path.join(saved_path, f'cell{"".join([str(i) for i in id_to_predict])}'))
    # data_stander.to_csv(os.path.join(saved_path, f'cell{"".join([str(i) for i in id_to_predict])}',
    #                                  'before_ss_data_2d.csv'), encoding='utf-8-sig')

    data_stander = pd.DataFrame(scaler.fit_transform(data_stander), index=data_stander.index,
                                columns=data_stander.columns)
    means = scaler.mean_  # scaler.data_min_ #
    stds = scaler.scale_  # scaler.data_max_ #

    # print(data_stander.head(10))

    # fig,axs = plt.subplots(len(data_stander.columns),1,figsize=(50,5*len(data_stander.columns)))
    # for i in range(len(data_stander.columns)):
    #     data_stander[data_stander.columns[i]].plot(ax=axs[i])
    #     axs[i].legend(loc='upper right')
    # print('-' * 10, '结束', '-' * 10)
    return data_stander.values, means, stds


def get_time_window(data, window):
    # print('-' * 10, '开始绘制时间窗', '-' * 10)
    num_samples, num_variabs = data.shape
    x_w = np.zeros((num_samples - window, window, num_variabs))
    y_w = np.zeros((num_samples - window, 1, num_variabs))
    for i in range(0, num_samples - window):
        x_w[i, :, :] = data[i:i + window, :]
        y_w[i, :, :] = data[i + window, :]
    # print('-' * 10, '结束', '-' * 10)
    return x_w, y_w


def can_receiver(data_length, interface="pcan", channel="PCAN_USBBUS1", bitrate=500000, sample_rate=10, window_size=10):
    """
    按指定采样频率采集 CAN 数据
    :return: 最新采样的浮点数（list）
    """
    bus = can.interface.Bus(interface=interface, channel=channel, bitrate=bitrate)
    # bus = can.interface.Bus(bustype="virtual", channel="test_channel", bitrate=500000)
    data_float = deque(maxlen=window_size)
    while True:
        data_iRow = []
        sample_interval = 1.0 / sample_rate
        for i in range(data_length):
            start_time = time.time()
            msg = bus.recv(timeout=1.5)
            if msg is None:
                print("未收到消息，超时")
                # 可以选择追加默认值或跳过，但这里建议等待消息
                continue  # 跳过本次循环，继续尝试接收
            try:
                data_iRow.append(struct.unpack(">f", msg.data)[0])
            except Exception as e:
                print("解码失败:", e)
            elapsed = time.time() - start_time
            # time.sleep(max(0, sample_interval - elapsed))
        data_float.append(np.array(data_iRow))

        # 如果累计到 window_size，就返回
        if len(data_float) == window_size:
            yield list(data_float)  # 返回拷贝，避免被修改



def get_CANData(bus, dataLength, sample_rate=10):
    # 读取CAN数据
    data_iRow = []
    sample_interval = 1.0 / sample_rate
    for i in range(dataLength):
        start_time = time.time()
        msg = bus.recv(timeout=1.0)
        if msg is None:
            print("未收到消息，超时")
            # 可以选择追加默认值或跳过，但这里建议等待消息
            continue  # 跳过本次循环，继续尝试接收
        try:
            data_iRow.append(struct.unpack(">f", msg.data)[0])
        except Exception as e:
            print("解码失败:", e)
        elapsed = time.time() - start_time
        time.sleep(max(0, sample_interval - elapsed))
    return np.array(data_iRow)

def get_index_CellsData(header, cell_num):
    # 根据表头得到每个单体对应数据列的索引
    res = []
    for cell_id in range(1, cell_num//2+1):
        Acell_id_data_featureStr = ['锂电池A单体' + f'{cell_id}电压', '蓄电池组A电压遥测', '蓄电池组A放电电流遥测',
                                    '蓄电池组A充电电流遥测', '蓄电池组A温度1', '蓄电池组A温度2', '蓄电池组A温度3']
        Bcell_id_data_featureStr = ['锂电池B单体' + f'{cell_id}电压', '蓄电池组B电压遥测', '蓄电池组B放电电流遥测',
                                    '蓄电池组B充电电流遥测', '蓄电池组B温度1', '蓄电池组B温度2', '蓄电池组B温度3']
        indicesA = [header.index(s) for s in Acell_id_data_featureStr if s in header]
        indicesB = [header.index(s) for s in Bcell_id_data_featureStr if s in header]
        res.append(indicesA)
        res.append(indicesB)
    return res

def get_index_Data2Java(header):
    indexPackFeatures_voltage = [header.index(s) for s in packFeatures_voltage if s in header]
    indexPackFeatures_current = [header.index(s) for s in packFeatures_current if s in header]
    indexPackFeatures_temp1 = [header.index(s) for s in packFeatures_temperature1 if s in header]
    indexPackFeatures_temp2 = [header.index(s) for s in packFeatures_temperature2 if s in header]
    indexPackFeatures_temp3 = [header.index(s) for s in packFeatures_temperature3 if s in header]
    indexFeatures_voltageA = [header.index(s) for s in features_voltageA if s in header]
    indexFeatures_voltageB = [header.index(s) for s in features_voltageB if s in header]
    return [indexPackFeatures_voltage, indexPackFeatures_current, indexPackFeatures_temp1, indexPackFeatures_temp2, indexPackFeatures_temp3, indexFeatures_voltageA, indexFeatures_voltageB]

def get_data(data, means, stds, device, cells_indexes):
    # fault2cell = {'单体SOH偏低':[7],'单体短路':[3], '单体间电压差过大':[2],'单体开路':[1],'单体气压过高':[5],'单体容量偏低':[9],
    #               '电池组短路':[2,4,6,8,10,12,14],
    #               '电池组开路':[1,3,5,7,9,11,13],
    #               '温度过高':[1,3,5,7,9,11,13],
    #               '泄露':[2,4,6,8,10,12,14]}
    # faultIdx = int(data[:,-1][0])
    # fault = fault_name_ch[faultIdx-1]
    # cells_Ids = fault2cell[fault]
    # 读取正常数据
    cells_data_stander_window_x = []
    cells_data_stander_window_y = []
    for i, cell_indexes in enumerate(cells_indexes):
        # if i+1 in cells_Ids:
            cell_data = data[:, cell_indexes]
            cell_data_stander = (cell_data - means[i]) / stds[i]
            cells_data_stander_window_y.append(cell_data_stander[-1])
            cell_data_stander = torch.tensor(cell_data_stander, dtype=torch.float32).to(device)
            cells_data_stander_window_x.append(cell_data_stander)
        # else:
        #     cells_data_stander_window_x.append(None)
        #     cells_data_stander_window_y.append(None)

    # return cells_data_stander_window_x, cells_data_stander_window_y, fault
    return cells_data_stander_window_x, cells_data_stander_window_y

def get_data2Java(data, allFeaturesIndexList):
    res = []
    for indexs in allFeaturesIndexList:
        res_i = []
        for index in indexs:
            res_i.append(data[index])
        res.append(res_i)
    return res


def processNormalData(dataNormalPath, window=10, cells_num=14, device='cpu'):
    # 读取正常数据
    data_Normal = pd.read_csv(dataNormalPath, encoding='utf-8-sig')
    header = data_Normal.columns.tolist()
    header.pop(0)   # 去掉时间戳
    means = []
    stds = []
    cells_data_Normal_stander_window_x = []
    cells_data_Normal_stander_window_y = []

    for id_to_predict in range(1, cells_num+1):
        if id_to_predict % 2 == 1:
            # A组电池组
            mono_voltage_name = '锂电池A单体' + f'{int(id_to_predict // 2 + 1)}电压'
            mono_voltage_Normal = data_Normal.loc[:, mono_voltage_name]

            voltage_name = ['蓄电池组A电压遥测']
            voltage_Normal = data_Normal.loc[:, voltage_name]

            current_name = ['蓄电池组A放电电流遥测', '蓄电池组A充电电流遥测']
            current_Normal = data_Normal.loc[:, current_name]

            temp_name = ['蓄电池组A温度1', '蓄电池组A温度2', '蓄电池组A温度3']
            temp_Normal = data_Normal.loc[:, temp_name]

        elif id_to_predict % 2 == 0:
            # B组电池组
            mono_voltage_name = '锂电池B单体' + f'{int(id_to_predict // 2)}电压'
            mono_voltage_Normal = data_Normal.loc[:, mono_voltage_name]
            voltage_name = ['蓄电池组B电压遥测']
            voltage_Normal = data_Normal.loc[:, voltage_name]
            current_name = ['蓄电池组B放电电流遥测', '蓄电池组B充电电流遥测']
            current_Normal = data_Normal.loc[:, current_name]
            temp_name = ['蓄电池组B温度1', '蓄电池组B温度2', '蓄电池组B温度3']
            temp_Normal = data_Normal.loc[:, temp_name]

        # '电压','放电电流','充电电流','温度1','温度2','温度3'
        cell_data_Normal = pd.concat([mono_voltage_Normal, voltage_Normal, current_Normal, temp_Normal], axis=1)
        # 归一化，得到正常数据的均值和方差
        cell_data_Normal_stander, means_mono, stds_mono = get_scaler(cell_data_Normal)
        # 对总数据进行归一化并划分时间窗
        cell_data_Normal_stander_window_x, cell_data_Normal_stander_window_y = get_time_window(cell_data_Normal_stander,window)
        cell_data_Normal_stander_window_x = torch.tensor(np.array(cell_data_Normal_stander_window_x),
                                                          dtype=torch.float32).to(device)
        # cell_data_Normal_stander_window_y = torch.tensor(np.array(cell_data_Normal_stander_window_y),dtype=torch.float32)
        cell_data_Normal_stander_window_y = np.array(cell_data_Normal_stander_window_y)


        cells_data_Normal_stander_window_x.append(cell_data_Normal_stander_window_x)
        cells_data_Normal_stander_window_y.append(cell_data_Normal_stander_window_y)
        # 若之后用不到均值和方差，再修改格式
        means.append(means_mono)
        stds.append(stds_mono)

    means = np.array(means)  # 各个cell的均值
    stds = np.array(stds)  # 各个cell的方差
    # 总数据，经过归一化后的总数据（模型输入），经过归一化后的总数据（真值），经过归一化后的正常数据（模型输入），经过归一化后的正常数据（真值），正常数据的均值，正常数据的方差
    return header, means, stds, cells_data_Normal_stander_window_x, cells_data_Normal_stander_window_y



def get_model_detect(model_name, device):
    models_path = ['Models/detect/' + f'cell{"".join(str(i))}/cell{"".join(str(i))}_{model_name}.pt' for i in
                   range(1, 15)]
    models = []
    for model_path in models_path:
        if os.path.exists(model_path):
            model = torch.jit.load(model_path).to(device)
            models.append(model)
        else:
            print(f'{model_path}不存在！')
    return models

def get_model_detect_out(modelsNamesStr, device):
    modelsNamesList = modelsNamesStr.split(",")
    cellsNum = len(modelsNamesList)
    models_path = ['Models/detect/' + f'cell{"".join(str(i+1))}/' + modelsNamesList[(i%2)*(cellsNum//2)+int(i/2)] for i in range(0, cellsNum)]
    models = []
    for model_path in models_path:
        if os.path.exists(model_path):
            model = torch.jit.load(model_path).to(device)
            models.append(model)
        else:
            print(f'{model_path}不存在！')
    return models



#
def get_model_diagnose(model_name, device):
    models_path = ['Models/diagnosis/' + f'{fault_eng}/{fault_eng}_{model_name}.pt' for fault_eng in fault_name_eng]
    models = []
    for model_path in models_path:
        if os.path.exists(model_path):
            model = torch.jit.load(model_path).to(device)
            models.append(model)
        else:
            print(f'{model_path}不存在！')
    return models

def get_model_diagnose_out(modelsNamesStr, device, diagnosisModelsFiles):
    modelsNamesList = modelsNamesStr.split(",")
    diagnosisModelsFilesList = diagnosisModelsFiles.split(",")
    models_path = ['Models/diagnosis/' + f'{fault_eng}/' + modelsNamesList[i] for i, fault_eng in enumerate(diagnosisModelsFilesList)]
    models = []
    for model_path in models_path:
        if os.path.exists(model_path):
            model = torch.jit.load(model_path).to(device)
            models.append(model)
        else:
            print(f'{model_path}不存在！')
    return models

def compute_thres(predValue, trueValue):
    # 完全使用 PyTorch 计算，避免 NumPy 兼容性问题
    pred_flat = predValue.detach().cpu().reshape(-1)
    true_flat = torch.tensor(trueValue.reshape(-1), dtype=torch.float32, device=predValue.device)
    mse = torch.mean((pred_flat - true_flat) ** 2)
    return mse.item()


def get_thres(model_detect, cell_Normal_data_x, cell_Normal_data_y):
    # 得到故障预警阈值
    model_detect.eval()
    pres = model_detect(cell_Normal_data_x)
    # print(pre.shape)
    # # 可视化
    # plt.figure(figsize=(10,5))
    # plt.plot(pres.detach().numpy().reshape(-1),label="预测值")
    # plt.plot(self.data_y.reshape(-1),label="真实值")
    # plt.legend(loc="upper right")
    # plt.xlabel('时间')
    # plt.title(f'cell{"".join([str(i) for i in self.id_to_predict])}')

    test_mse = mean_squared_error(
        cell_Normal_data_y.reshape(-1), pres.detach().cpu().numpy().reshape(-1))
    test_mae = mean_absolute_error(
        cell_Normal_data_y.reshape(-1), pres.detach().cpu().numpy().reshape(-1))
    test_r2 = r2_score(
        cell_Normal_data_y.reshape(-1), pres.detach().cpu().numpy().reshape(-1))
    test_evs = explained_variance_score(
        cell_Normal_data_y.reshape(-1), pres.detach().cpu().numpy().reshape(-1))
    # thres = torch.quantile((pres.detach().reshape(-1)-self.data_y.reshape(-1))**2, q=0.95).numpy()
    # thres = ((pres.detach().reshape(-1) - cell_Normal_data_y.reshape(-1)) ** 2).mean().numpy()
    thres = compute_thres(pres, cell_Normal_data_y)

    return pres, thres, test_mse, test_mae, test_r2, test_evs


HOST = "localhost"
PORT = 12347  # 选择一个未被占用的端口
HOST_CAN = "127.0.0.1"
PORT_CAN = 20000
ROLE = "diagnose"
import select

OneSecondSendData = 1
# def handle_client(conn):
#     try:
#         r, _, _ = select.select([conn], [], [], 0.05)  # 0.5秒检查一次
#         if r:
#             data = conn.recv(1024)
#             if not data:
#                 print("客户端关闭连接", file=sys.stderr)
#                 return True

#             msg = data.decode('utf-8').strip()
#             if msg.upper() == "STOP":
#                 print("收到停止信号", file=sys.stderr)
#                 return True
#         return False
#     except Exception as e:
#         print(f"处理客户端时发生错误: {e}", file=sys.stderr)
#         return True

# def listen_for_stop(conn):
#     global stop_flag
#     while not stop_flag:
#         try:
#             if handle_client(conn):
#                 stop_flag = True
#                 break
#         except Exception as e:
#             print(f"监听线程错误: {e}", file=sys.stderr)
#             stop_flag = True
#             break
#         time.sleep(0.1)  # 添加短暂休眠减少CPU占用

# def ensure_can_server():
#     while True:
#         try:
#             s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             s.connect((HOST_CAN, PORT_CAN))
#             s.sendall(ROLE.encode())
#             print("[Detect] Connected to CANHub.")
#             return s
#         except ConnectionRefusedError:
#             print("[Detect] CANHub not running, try it again...")
#             pass


# def recv_exact(sock, n):
#     data = b""
#     while len(data) < n:
#         packet = sock.recv(n - len(data))
#         if not packet:
#             return None
#         data += packet
#     return data

def get_statys_change_nums(status, return_result, cell_id, PredFault_ing, abnormalData):
    if status[cell_id - 1] != return_result:

        if return_result != 0:
            abnormalData += 1
            PredFault_ing[cell_id - 1][return_result-1] += 1
    status[cell_id - 1] = return_result
    #   虚警


    return PredFault_ing, abnormalData

def getMeansAndStd(means, stds):
    means_list = []
    stds_list = []
    means = np.array(means)
    stds = np.array(stds)
    means_list.append(means[0, 1:].tolist())
    means_list.append(means[1, 1:].tolist())
    means_list.append(means[::2, 0].tolist())
    means_list.append(means[1::2, 0].tolist())
    means_list = [item for sublist in means_list for item in sublist]
    stds_list.append(stds[0, 1:].tolist())
    stds_list.append(stds[1, 1:].tolist())
    stds_list.append(stds[::2, 0].tolist())
    stds_list.append(stds[1::2, 0].tolist())
    stds_list = [item for sublist in stds_list for item in sublist]
    return means_list, stds_list

def getDataset(data_path, window=10):
    cellsData = pd.read_csv(data_path, encoding='utf-8-sig').iloc[:, 1:-1].values
    cell_data_window_x, cell_data_window_y = get_time_window(cellsData, window)
    # 总数据，经过归一化后的总数据（模型输入），经过归一化后的总数据（真值），经过归一化后的正常数据（模型输入），经过归一化后的正常数据（真值），正常数据的均值，正常数据的方差
    return cell_data_window_x


def getDataLiter(data):
    i = 0
    dataLength = len(data)

    while not stop_flag:
        time.sleep(1)
        data_epoch = data[i]
        i += 1
        i = i % dataLength
        yield data_epoch

stop_flag = False  # 全局标志位

def process_new_data(myargs, device):
    """处理新数据的流程"""
    print("使用新数据处理流程...")
    
    # 加载新数据
    data_folder = myargs.data_path_Normal
    volt_tensors, tem_tensors, current_list, time_list = load_new_data_tensors(data_folder)
    
    # 加载新数据模型
    new_data_model = load_new_data_model(device=device)
    
    # 离散化充放电倍率
    rate_ch = (10*current_list[:, 0]).long()
    rate_dch = (10*current_list[:, 1]).long()
    
    # 构建输入特征
    input_features = torch.stack([tem_tensors[:, :-5], volt_tensors[:, :-5]], dim=2)
    
    # 目标为未来5个时刻
    target = torch.stack([tem_tensors[:, -5:], volt_tensors[:, -5:]], dim=2)
    
    # 模拟检测过程
    detect_dict = {}
    batch = 1
    
    print("开始新数据检测...")
    for i in range(min(10, len(input_features))):  # 只处理前10个样本作为示例
        x_seq = input_features[i:i+1].to(device)
        ch = rate_ch[i:i+1].to(device)
        dch = rate_dch[i:i+1].to(device)
        
        with torch.no_grad():
            y_pred = new_data_model(x_seq, ch, dch)
            y_true = target[i:i+1].to(device)
            
            # 计算预测误差
            mse = torch.mean((y_pred - y_true) ** 2).item()
            
            # 简单的阈值判断（可以根据实际情况调整）
            threshold = 0.01  # 示例阈值
            if mse > threshold:
                detect_dict[f"新数据样本_{batch}"] = 1
                print(f"样本 {batch}: 检测到异常 (MSE: {mse:.6f})")
            else:
                detect_dict[f"新数据样本_{batch}"] = 0
                print(f"样本 {batch}: 正常 (MSE: {mse:.6f})")
        
        batch += 1
    
    # 保存结果
    df_detect_dict = pd.DataFrame(list(detect_dict.items()), columns=("sample", "fault"))
    print(df_detect_dict)
    df_detect_dict.to_csv("output_new_data_detect.csv", index=True, encoding="utf-8")
    print("新数据检测完成，结果已保存到 output_new_data_detect.csv")
    
    return 0


def process_old_data(myargs, device):
    """处理老数据的流程（原有逻辑）"""
    print("使用老数据处理流程...")
    
    window = myargs.window
    data_path_Normal = myargs.data_path_Normal

    cells_num = 14
    header, means, stds, cells_data_Normal_stander_window_x, cells_data_Normal_stander_window_y = processNormalData(data_path_Normal, window, cells_num, device)
    features_len = len(header)

    cells_indexes = get_index_CellsData(header, cells_num)
    allFeaturesIndexList = get_index_Data2Java(header)

    # 加载模型
    detect_models = get_model_detect(myargs.detect_model_name, device)

    cell_thres = []
    for cell_id in range(1, cells_num + 1):
        pres, thres, test_mse, test_mae, test_r2, test_evs = get_thres(detect_models[cell_id - 1],
                                                                       cells_data_Normal_stander_window_x[cell_id - 1],
                                                                       cells_data_Normal_stander_window_y[cell_id - 1])
        cell_thres.append(thres)
    means2Java, stds2Java = getMeansAndStd(means, stds)

    initData2Java = {"means": means2Java,
                     "stds": stds2Java,
                     "features": features}

    PredFault_ing = np.zeros((cells_num, len(fault_name_eng)))
    abnormalCount = 0
    batch = 1

    detect_dict = {}
    overrunNum_cells = np.zeros(cells_num)
    upOverrunNum = int(1.5 * window)

    # xie todo: 老数据处理流程
    myDataSet = getDataset("Data/combined_data_20.csv", window=window)
    try:
        # 数据预处理
        for cells_data_window_list in getDataLiter(myDataSet):
            if len(cells_data_window_list) != 0:
                cells_data_window = np.array(cells_data_window_list)
                cells_data2Java = get_data2Java(cells_data_window_list[-1], allFeaturesIndexList)
                cells_data_stander_window_x, cells_data_stander_window_y = get_data(cells_data_window, means, stds, device, cells_indexes)

                cells_pred_fault = []
                for cell_id in range(1, cells_num+1):
                    detect_model = detect_models[cell_id - 1].eval()
                    cell_data_stander_window_input = cells_data_stander_window_x[cell_id - 1].unsqueeze(0)
                    cell_data_stander_window_trueValue = cells_data_stander_window_y[cell_id - 1]
                    cell_data_stander_window_predValue = detect_model(cell_data_stander_window_input)
                    
                    cell_str = f"单体电池A{cell_id // 2 + 1}_{batch}" if cell_id % 2 != 0 else f"单体电池B{cell_id // 2}_{batch}"
                    thres_Error = compute_thres(cell_data_stander_window_predValue, cell_data_stander_window_trueValue)
                    
                    if thres_Error > cell_thres[cell_id - 1]:
                        overrunNum_cells[cell_id - 1] += 1
                        if overrunNum_cells[cell_id - 1] >= upOverrunNum:
                            detect_dict[cell_str] = 1
                            overrunNum_cells[cell_id - 1] = 0
                            cells_pred_fault.append("异常")
                        else:
                            detect_dict[cell_str] = 0
                            cells_pred_fault.append("正常")
                    else:
                        detect_dict[cell_str] = 0
                        cells_pred_fault.append("正常")

                cells_pred_fault = np.array(cells_pred_fault)
                cells_pred_fault2Java = cells_pred_fault[::2].tolist() + cells_pred_fault[1::2].tolist()
                features2Java = [packFeatures_voltage, packFeatures_current, packFeatures_temperature1, packFeatures_temperature2, packFeatures_temperature3, features_voltageA, features_voltageB]
                PredFault_ing_ranged = np.concatenate((PredFault_ing[::2,:], PredFault_ing[1::2,:]), axis=0).tolist()

                data2Java = {"cells_id": [f"单体电池A{cell_id}" for cell_id in range(1, cells_num//2+1)]+[f"单体电池B{cell_id}" for cell_id in range(1, cells_num//2+1)],
                             "pred_fault": cells_pred_fault2Java,
                             "fault_name_ch": fault_name_ch,
                             "PredFault_ing": PredFault_ing_ranged,
                             "features": features2Java,
                             "cell_data_trans": cells_data2Java,
                             "abnormalCount": abnormalCount,
                             "status": "valid",
                             "batch": f"{batch}"
                             }
                batch += 1
            else:
                data2Java = {"status": "invalid"}


    except KeyboardInterrupt:
        print("用户中断")
    finally:
        print("老数据处理完成")
        df_detect_dict = pd.DataFrame(list(detect_dict.items()), columns=["cell", "fault"])
        print(df_detect_dict)
        df_detect_dict.to_csv("output_detect_dict.csv", index=True, encoding="utf-8")

    return 0


def main(myargs):
    """主函数：自动判断数据类型并选择相应的处理流程"""
    # 跟Java通信
    # with (socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s):
    #     status = []*7
    #     s.bind((HOST, PORT))
    #     s.listen()
    #     conn, addr = s.accept()  # 等待 Java 连接
    # 加载数据

    # can_sock = ensure_can_server()
    window = myargs.window
    device = myargs.device

    data_path_Normal = myargs.data_path_Normal
    
    # xie todo: 自动判断数据类型并选择相应的处理流程
    data_type = detect_data_type(data_path_Normal)
    print(f"检测到数据类型: {data_type}")
    
    if data_type == 'new_data':
        # 新数据处理流程
        return process_new_data(myargs, device)
    else:
        # 老数据处理流程（原有逻辑）
        return process_old_data(myargs, device)


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <json_parameters>", file=sys.stderr)
    #     sys.exit(1)
    try:
        parser = argparse.ArgumentParser(description='PyTorch Cellular Automata')
        # parser.add_argument('--port', type=int, default=12345,
        #                     help='TCP port for Java socket client')
        # 读取的数据文件名
        # parser.add_argument('--data_path', type=str, default="Data/combined_data_200.csv",
        #                     help='path to dataset')
        # 读取故障预警模型名
        parser.add_argument('--detect_model_name', type=str, default='AutoEncoder')
        # parser.add_argument('--selectedWarningModels', type=str)
        # 读取故障诊断模型名
        # parser.add_argument('--diagnose_model_name', type=str, default='LinerNet')
        # parser.add_argument('--selectedDiagnosisModels', type=str)
        # parser.add_argument('--diagnosisModelsFiles', type=str)
        # 读取设备
        parser.add_argument('--device', type=str, default='cpu')

        parser.add_argument('--data_path_Normal', type=str, default='20240118锂电测试数据（内部）_10000.csv',
                            help='path to Normal dataset')
        parser.add_argument('--window', type=int, default=10,
                            help='window size')
#         parser.add_argument('--sample_rate', type=int, default=10,
#                             help='sample rate')

        args = parser.parse_args()
        a = main(args)
        sys.exit(a)
    except Exception as e:
        print(f"Error: {e}")
    # except json.JSONDecodeError:
    #     print("错误：参数必须是 JSON 格式", file=sys.stderr)
    #     sys.exit(1)
