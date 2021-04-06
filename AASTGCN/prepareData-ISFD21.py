import os
import zipfile
import numpy as np
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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
    if dataset == 'SSFD21':
        # SSFD-DATA-110 Nodes
        ###########SSFD Sector##############
        # Sector1
        # Code_list = ['APD','BBL','BHP','CTA-PB','ECL','LIN','RIO','SCCO','SHW','VALE'] # 10
        # Sector2
        # Code_list = ['ATVI','BIDU','CMCSA','DIS','GOOG','NFLX','NTES','T','TMUS','VZ'] # 10
        # Sector3
        # Code_list = ['AMZN','BKNG','HD','LOW','MCD','MELI','NKE','SBUX','TM','TSLA'] # 10
        # Sector4
        # Code_list = ['BUD','COST','DEO','EL','KO','PEP','PG','PM','UL','WMT'] # 10
        # Sector5
        # Code_list = ['BP','COP','CVX','ENB','EQNR','PTR','RDS-B','SNP','TOT','XOM'] # 10
        # Sector6
        # Code_list = ['BAC','BML-PG','BML-PL','BRK-B','JPM','LFC','MA','MS','V','WFC'] # 10
        # Sector7
        # Code_list = ['ABT','JNJ','LLY','MDT','MRK','NVO','NVS','PFE','TMO','UNH'] # 10
        # Sector8
        # Code_list = ['BA','CAT','DE','GE','HON','LMT','MMM','RTX','UNP','UPS'] # 10
        # Sector9
        # Code_list = ['AMT','CSGP','DLR','EQIX','PLD','SBAC','SPG','SPG-PJ','WELL','WY'] # 10
        # Sector10
        # Code_list = ['AAPL','ADBE','ASML','AVGO','CRM','CSCO','INTC','MSFT','NVDA','TSM'] # 10
        # Sector11
        # Code_list = ['AEP','D','ES','EXC','NEE','NGG','PEG','SO','SRE','XEL'] # 10
        # # Code_list = file_name('../data/SHY21-Arima-back')

        ##############################
        Code_list = file_name('./data/SSFD21-Arima')

        ###############################
        data_path = os.path.join('./data/SSFD21/SSFD-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            # print(code)
            df_code = df[(df['Code'] == code)]
            df_code_adj_close = df_code['Adj Close'].values.tolist()
            df_group_data[code] = df_code_adj_close
            df_code_label = df_code['Label'].values.tolist()
            df_label_data[code] = df_code_label
        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)
        data = np.transpose([df_group_data.values])
    elif dataset == 'ISFD21':
        # SSFD-DATA-105 Nodes
        ##########
        # Sector1
        # Code_list = ['AVD','CF','CTA-PA','CTA-PB','FMC','ICL','IPI','MOS','SMG'] # 9
        # Sector2
        # Code_list = ['AMOV','AMX','BCE','CHT','ORAN','T','TMUS','TU','VOD','VZ'] # 10
        # Sector3
        # Code_list = ['ALV','BWA','DAN','DORM','GNTX','GT','LEA','LKQ','MGA','VC'] # 10
        # Sector4
        # Code_list = ['ADM','ALCO','BG','CALM','CHSCP','FDP','IBA','LMNR','TSN'] # 9
        # Sector5
        # Code_list = ['CEO','CLR','CNQ','COP','DVN','EOG','HES','MRO','OXY','PXD'] # 10
        # Sector6
        # Code_list = ['AMP','BAM','BEN','BK','BLK','BX','KKR','NTRS','STT','TROW'] # 10
        # Sector7
        # Code_list = ['CERN','CPSI','HMSY','HSTM','MDRX','NXGN','OMCL'] # 7
        # Sector8
        # Code_list = ['AIT','DXPE','FAST','GWW','LAWS','MSM','PKOH','SYX','WCC','WSO'] # 10
        # Sector9
        # Code_list = ['CBRE','CIGI','CSGP','CSR','FRPH','IRCP','JLL','KW','NTP','TCI'] # 10
        # Sector10
        # Code_list = ['ADSK','ANSS','CDNS','CRM','CTXS','INTU','PTC','SAP','SSNC','TYL'] # 10
        # Sector11
        # Code_list = ['AEP','DTE','DUK','ED','ES','NEE','PCG','SO','WFC','XEL'] #10

        ##########
        # Code_list = ['CDNS','CTXS','PTC','SAP','SSNC']
        # Code_list = ['ADSK','ANSS','CRM','INTU','TYL']
        #########################################
        Code_list = file_name('./data/ISFD21-Arima')
        data_path = os.path.join('./data/ISFD21/ISFD-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            df_code = df[(df['Code'] == code)]
            df_code_adj_close = df_code['Adj Close'].values.tolist()
            df_group_data[code] = df_code_adj_close
            df_code_label = df_code['Label'].values.tolist()
            df_label_data[code] = df_code_label

        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)

        data = np.transpose([df_group_data.values])
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    # Normalization using Z-score method
    X = data
    print('X.shape:{}'.format(X.shape))
    # means = np.mean(X, axis=(0, 2))
    means = np.mean(X, axis=(0, 1))

    X = X - means.reshape(1, -1, 1)
    # stds = np.std(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 1))
    X = X / stds.reshape(1, -1, 1)

    means = means.reshape(1,1,1,1)
    stds = stds.reshape(1,1,1,1)
    print('means:{}'.format(means))
    print('stds:{}'.format(stds))
    return X,means,stds,data

def compute_Euclidean_Distance(vector1,vector2):
    op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
    # op2 = np.linalg.norm(vector1 - vector2)
    return op1

def generate_adj():
    X = load_st_dataset('ISFD21')
    # print('X.shape:{}'.format(X.shape))
    # step1 Generate the rise and fall sequence of each stock based on the training set data
    # Get training data
    code_list = file_name('./data/ISFD21-Arima')
    test_size = int(X.shape[0] * 0.2)
    train_size = X.shape[0] - test_size
    total_data = pd.read_csv('./data/ISFD21/ISFD-V1_11.csv')
    diff_list = []
    for code in code_list:
        # print('code:{}'.format(code))
        # Get the first-order difference data of each stock (training part)
        stock_data = total_data[total_data['Code']==code][:train_size]['Diff'].values
        # print(stock_data)
        diff_list.append(stock_data)
    # print('diff_list.shape:{}'.format(len(diff_list)))

    # step2 Calculate the similarity of each stock's rise and fall sequence Use the similarity results
    #       to construct a 105*105 static adjacency matrix

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