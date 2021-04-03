import os
import numpy as np
import argparse
import configparser
import pandas as pd
import torch

DEVICE = torch.device('cuda:0')

# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./configurations/ISFD21_aastgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
# points_per_hour = int(data_config['points_per_hour']) # Abandoned
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
# num_of_weeks = int(training_config['num_of_weeks']) # Abandoned
# num_of_days = int(training_config['num_of_days']) # Abandoned
# num_of_hours = int(training_config['num_of_hours'])  # Abandoned



# get file name list in file_dir
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

    # Normalization using my Z-score method in main.py
    # X = data
    # return X

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
               in range(X.shape[0] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[i: i + num_timesteps_input,:,:])
        target.append(X[i + num_timesteps_input: j,:,0])

    # return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))
    return torch.tensor([np.array(feature) for feature in features],dtype=torch.float),torch.tensor([np.array(targe) for targe in target],dtype=torch.float)


def load_graphdata_channel_stp(Data_name,num_timesteps_input,num_timesteps_output, DEVICE, batch_size, shuffle=True):
    """
    Final Data prepare function for AASTGCN / ASTGCN
    :return:
    """
    # # load data-ISFD21
    X, means, stds, data = load_st_dataset(Data_name)
    print('X.shape:{}'.format(X.shape)) # X.shape:(2516, 105, 1)
    # print(X)

    split_line1 = int(X.shape[0] * 0.6)
    split_line2 = int(X.shape[0] * 0.8)
    train_original_data = X[:split_line1,:,:]

    val_original_data = X[split_line1:split_line2,:,:]
    test_original_data = X[split_line2:,:,:]

   # train valid test #DONE
    train_x, train_target = generate_dataset(train_original_data,
                                                      num_timesteps_input=num_timesteps_input,
                                                      num_timesteps_output=num_timesteps_output)
    train_x = train_x.permute((0,2,3,1))
    train_target = train_target.permute((0,2,1))
    print('trainx.shape:{}'.format(train_x.shape))
    print('train_target.shape:{}'.format(train_target.shape))

    val_x, val_target = generate_dataset(val_original_data,
                                                      num_timesteps_input=num_timesteps_input,
                                                      num_timesteps_output=num_timesteps_output)
    val_x =  val_x.permute((0,2,3,1))
    val_target = val_target.permute((0,2,1))
    print('val_x.shape:{}'.format(val_x.shape))
    print('val_target.shape:{}'.format(val_target.shape))
    print('val_x.shape:{}'.format(val_x.shape))
    print('val_target.shape:{}'.format(val_target.shape))

    test_x, test_target = generate_dataset(test_original_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=num_timesteps_output)
    test_x = test_x.permute((0, 2, 3, 1))
    test_target = test_target.permute((0, 2, 1))
    print('test_x.shape:{}'.format(test_x.shape))
    print('test_target.shape:{}'.format(test_target.shape))
    print('test_x.shape:{}'.format(test_x.shape))
    print('test_target.shape:{}'.format(test_target.shape))

    # means stds #DONE
    print('means.shape:{}'.format(means.shape))
    print('stds.shape:{}'.format(stds.shape))

    # torch 4 float to avoid error
    train_x = np.array(train_x)
    train_target = np.array(train_target)

    val_x = np.array(val_x)
    val_target = np.array(val_target)

    test_x = np.array(test_x)
    test_target = np.array(test_target)


    # # load ISFD21-adj_matrix
    # adj_mx = np.load('./data/ISFD21_adj.npy') # static_adj_matrix with history price data

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, means, stds

# load_graphdata_channel_stp(num_timesteps_input=12, num_timesteps_output=12,DEVICE=DEVICE, batch_size=64)