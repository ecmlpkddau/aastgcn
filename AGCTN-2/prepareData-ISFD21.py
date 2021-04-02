# 20210310
# 支持按照METR-LA的数据格式，载入ISFD21数据 #DONE utils_ISFD21.py

# 定义ISFD21数据的邻接矩阵，为了引入空间信息，暂时将股票距离作为邻接矩阵的计算依据 #DONE utils_ISFD21.py
    # 根据股票的历史价格数据构建股票的邻接矩阵，有一些需要注意的点
        # 1. 构建邻接矩阵时不能涉及到未来信息，即只用训练数据构建邻接矩阵
        # 2. 原始STGCN邻接矩阵是静态的。如果引入动态的邻接矩阵是否会因为股票的随机特性引起模型异常？
            # 反过来，原始的静态邻接矩阵是否能反映出交通路网的动态变化特征？
                # 方案1： 根据股票的历史价格构建静态邻接矩阵（根据涨跌）#DONE utils_ISFD21.py
                # 每支股票的adj_close 对应着一个涨跌序列构成，根据股票的涨跌序列（实际上也是一阶差分）计算序列之间的相似度构建邻接矩阵

# 新增早停机制并存储最优模型,支持模型的载入 #DONE main.py
    #回顾了一下早停策略，到这里也发现了之前的部分模型早停策略存在漏洞，正常应该是当触发早停机制时进行如下操作：
        # 1. 停止训练
        # 2. 载入PATIENCE前的模型参数为最优
        # 之前只进行了第一步，导致模型可能不是最优的。
# 新增测试部分，以及结果的输出,可视化 #DONE main.py
# 新增预测结果指标评价 #DONE main.py

# 20210311
# 支持结果的检验，即针对每支股票计算对应的评估指标，而不是基于全局结果 #DONE main.py test.py
    # 结果果然由问题，原始的数据还原方法不可用
        # 支持结果的正确还原 #DONE main.py

# 支持GPU运行程序

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

# A, X, means, stds = load_metr_la_data()
# A.shape:(207,207) (nodes_num,nodes,num)
# X.shape:(207,2,34272) (nodes_num,features_num,data_length)
# print('original A:')
# print(A)
# A = get_normalized_adj(A)
# print("normalized A:")
# print(A)
# X_data.shape:(34253, 207, 10, 2) (data_length,nodes_num,input_timestep,features_num)
# Y_data.shape:(34253,207,10) (data_length, nodes_num,output_timestep)

# 获取指定文件夹目录下的所有文件名称（不包括扩展名）
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
        # 矩阵转置，横向取数据
        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)
        # df_label_data = pd.DataFrame(df_label_data.values.T,index=df_label_data.columns,columns=df_label_data.index)
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

    # Normalization using my Z-score method in main.py
    # X = data
    # return X


# # For ISFD21 nodes_num = 105 features=1 length=2518  X.shape(105,1,2516)
# X = load_st_dataset('ISFD21')
# X = torch.from_numpy(X)
# X = X.permute(1,2,0)
# print('X.shape:{}'.format(X.shape)) # (nodes_num,features_num,data_length) (207,2,34272)
# X_data, Y_data = generate_dataset(X,num_timesteps_input=10,num_timesteps_output=10)
# print('X_data.shape:{}'.format(X_data.shape))
# print(X_data)
# print('Y_data.shape:{}'.format(Y_data.shape))
# print(Y_data)




# 计算两个向量的“空间”距离
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

# 生成静态邻接矩阵 road:'./data/ISFD21_adj.npy'
# generate_adj()
# 载入邻接矩阵
# adj = np.load('./data/ISFD21_adj.npy')
# print('adj:{}'.format(adj.shape))
# print(adj)

# # For ISFD21 nodes_num = 105 features=1 length=2518  X.shape(105,1,2516)


# get X,A,means,stds
#
# X,means,stds = load_st_dataset('ISFD21')
# print(means.shape)
# print(stds.shape)

# X = load_st_dataset('ISFD21') # load data

# the flow to z-score the data (recover data is the same flow)
#  step1: reshape for StandardScaler
# X = np.reshape(X,(-1,X.shape[1]))
# ss = StandardScaler()

# step2: Z-score normalize
# std_data = ss.fit_transform(X)

# step3: recover shape for train
# X = std_data
# X = X[:,:,None]

# step4: total review for transform and inverse_transform

# # data normalized
# ss = StandardScaler()
# X = ss.fit_transform(np.reshape(X,(-1,X.shape[1])))[:,:,None]
# print('X.shape:{}'.format(X.shape))
# # data unnormalized
# ori_data = ss.inverse_transform(np.reshape(X,(-1,X.shape[1])))[:,:,None]
# print('ori_data.shape:{}'.format(ori_data.shape))


# X = torch.from_numpy(X)
# X = X.permute(1,2,0)

# load static adj
# adj = np.load('./data/ISFD21_adj.npy')


# just for test
# print('X.shape:{}'.format(X.shape)) # (nodes_num,features_num,data_length) (207,2,34272)
# X_data, Y_data = generate_dataset(X,num_timesteps_input=10,num_timesteps_output=10)
# print('X_data.shape:{}'.format(X_data.shape))
# print(X_data)
# print('Y_data.shape:{}'.format(Y_data.shape))
# print(Y_data)

num_timesteps_input = 12
num_timesteps_output = 12
X, means, stds, data = load_st_dataset('ISFD21')

# test_size = int(X.shape[0] * 0.2)
# train_size = X.shape[0] - test_size
X = torch.from_numpy(X)
X = X.permute(1, 2, 0)

# load ISFD21-adj_matrix
# A = np.load('./data/ISFD21_adj.npy') # static_adj_matrix with history price data

# split data 6:2:2
split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.8)
# print('test_length:{}'.format(X.shape[2]-split_line2))

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]

# change for 0.8 training data and 0.2 testing data
# train_original_data = X[:, :, :train_size]
# test_original_data = X[:, :, train_size:]
training_input, training_target = generate_dataset(train_original_data,
                                                   num_timesteps_input=num_timesteps_input,
                                                   num_timesteps_output=num_timesteps_output)
training_input = np.array(training_input)
training_input = training_input.transpose((0,1,3,2))
training_target = np.array(training_target)

# training_target = np.array(training_target).transpose((0,1,3,2))
print('training_input.shape:{}'.format(training_input.shape))
print('training_target.shape:{}'.format(training_target.shape))
# print(training_input.shape) # torch.Size([20549, 207, 12, 2]) (num_samples, num_vertices, num_timesteps_input, num_features). num_features = 2
# print(training_input)
# print('training_target:')
# print(training_target.shape) # torch.Size([20549, 207, 3]) (num_samples, num_vertices, num_timesteps_output, num_features). num_features = 1
# print(training_target)

val_input, val_target = generate_dataset(val_original_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=num_timesteps_output)

val_input = np.array(val_input)
val_input = val_input.transpose((0,1,3,2))
val_target = np.array(val_target)

test_input, test_target = generate_dataset(test_original_data,
                                           num_timesteps_input=num_timesteps_input,
                                           num_timesteps_output=num_timesteps_output)

test_input = np.array(test_input)
test_input = test_input.transpose((0,1,3,2))
test_target = np.array(test_target)


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

norm_dic,norm_train,norm_val,norm_test = normalization(training_input,val_input,test_input)
# print(norm_train)
# print(norm_train.shape)
print(norm_dic['_mean'])
print(norm_dic['_std'])