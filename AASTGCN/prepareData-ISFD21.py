import os
import zipfile
import numpy as np
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def load_metr_la_data():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds
def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave
def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[:, :, i: i + num_timesteps_input].permute(0, 2, 1))
        target.append(X[:, 0, i + num_timesteps_input: j])

    # return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))
    return torch.tensor([np.array(feature) for feature in features],dtype=torch.float),torch.tensor([np.array(targe) for targe in target],dtype=torch.float)



def file_name(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file = os.path.splitext(file)[0]
            file_list.append(file)
    file_list.sort()
    return file_list
def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'ACL18':
        Code_list = ['AEP', 'D', 'DUK', 'EXC', 'NEE', 'NGG', 'PCG', 'PPL', 'SO', 'SRE']

        data_path = os.path.join('../data/ACL18/ACL-V1_2.csv')
        # Code_list = get_target_stock_id_set(data_path)
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        len_count = 0
        for code in Code_list:
            print(code)
            if code != 'GMRE':
                df_code = df[(df['Code'] == code) & (df['Date'] >= '2014-01-01') & (df['Date'] <= '2016-01-01')]
                len_df_code = len(df_code)
                if len_df_code == 504:
                    len_count += 1
                    df_code_adj_close = df_code['Adj Close'].values.tolist()
                    print('data_len:')
                    print(len(df_code_adj_close))
                    df_group_data[code] = df_code_adj_close
                    df_code_label = df_code['Label'].values.tolist()
                    df_label_data[code] = df_code_label
        # 矩阵转置，横向取数据
        print('right count:')
        print(len_count)
        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)
        # df_label_data = pd.DataFrame(df_label_data.values.T,index=df_label_data.columns,columns=df_label_data.index)
        data = np.transpose([df_group_data.values])
    elif dataset == 'SHY21':
        # 载入SHY21数据110(不包括以下8个后加的股票（有一个原始数据集中有）)
        Code_list = ['AVD','CF','CTA-PA','CTA-PB','FMC','ICL','IPI','MOS','SMG']
        # Code_list = file_name('../data/SHY21-Arima')
        # Code_list = file_name('../data/SHY21-Arima-back')
        data_path = os.path.join('../data/ACL18/SHY-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            print(code)
            if code != 'GMRE':
                df_code = df[(df['Code'] == code)]
                df_code_adj_close = df_code['Adj Close'].values.tolist()
                df_group_data[code] = df_code_adj_close
                df_code_label = df_code['Label'].values.tolist()
                df_label_data[code] = df_code_label

        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)

        data = np.transpose([df_group_data.values])
    elif dataset == 'ISFD21':
        # 载入ISFD21数据105
        # Sector1
        # Code_list = ['AVD','CF','CTA-PA','CTA-PB','FMC','ICL','IPI','MOS','SMG']
        # Sector2
        # Code_list = ['AMOV','AMX','BCE','CHT','ORAN','T','TMUS','TU','VOD','VZ']
        # Sector10
        # Code_list = ['ADSK','ANSS','CDNS','CRM','CTXS','INTU','PTC','SAP','SSNC','TYL']
        # Code_list = ['CDNS','CTXS','PTC','SAP','SSNC']
        # Code_list = ['ADSK','ANSS','CRM','INTU','TYL']

        Code_list = file_name('./data/ISFD21-Arima')
        data_path = os.path.join('./data/ISFD21/ISFD-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            # print(code)
            if code != 'GMRE':
                df_code = df[(df['Code'] == code)]
                df_code_adj_close = df_code['Adj Close'].values.tolist()
                df_group_data[code] = df_code_adj_close
                df_code_label = df_code['Label'].values.tolist()
                df_label_data[code] = df_code_label
        # 矩阵转置，横向取数据
        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)
        # df_label_data = pd.DataFrame(df_label_data.values.T,index=df_label_data.columns,columns=df_label_data.index)
        data = np.transpose([df_group_data.values])
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    # Normalization using Z-score method
    X = data
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)
    return X,means,stds,data

def compute_Euclidean_Distance(vector1,vector2):
    op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
    # op2 = np.linalg.norm(vector1 - vector2)
    return op1

def generate_adj():
    X = load_st_dataset('ISFD21')
    # print('X.shape:{}'.format(X.shape))
    # step1 根据训练集数据生成每支股票的涨跌序列
    # 获取训练集数据
    code_list = file_name('./data/ISFD21-Arima')
    test_size = int(X.shape[0] * 0.2)
    train_size = X.shape[0] - test_size
    total_data = pd.read_csv('./data/ISFD21/ISFD-V1_11.csv')
    diff_list = []
    for code in code_list:
        # print('code:{}'.format(code))
        # 拿到每支股票（训练部分）的一阶差分数据
        stock_data = total_data[total_data['Code']==code][:train_size]['Diff'].values
        # print(stock_data)
        diff_list.append(stock_data)
    # print('diff_list.shape:{}'.format(len(diff_list)))

    # step2 计算每支股票涨跌序列相似度 利用相似度结果构建105*105的静态邻接矩阵

    # method-1 cosine-similarity
    adj_matrix = cosine_similarity(diff_list)
    # print('cosine_adj_matrix.shape:{}'.format(adj_matrix.shape))
    # print(adj_matrix)

    # # method-2 Euclidean_Distance
    # distance_list = []
    # for i in range(len(diff_list)):
    #     for j in range(len(diff_list)):
    #         dis_value = compute_Euclidean_Distance(diff_list[i],diff_list[j])
    #         distance_list.append(dis_value)
    # adj_matrix = np.array(distance_list).reshape(len(diff_list),len(diff_list))
    # print('distance_adj_matrix.shape:{}'.format(distance_array.shape))
    # print(distance_array)
    np.save('./data/ISFD21_adj.npy',adj_matrix)
    print('ISFD21 Adj matrix save succcess! ')
    return adj_matrix

def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    print('train.shape:{}'.format(train.shape))
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    mean = train.mean(axis=(0,3), keepdims=True)
    std = train.std(axis=(0,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('mean:{}'.format(mean))
    print('std.shape:',std.shape)
    print('std:{}'.format(std))

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm