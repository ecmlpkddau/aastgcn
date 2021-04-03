# aastgcn
Code and Data for ECMLPKDD2021 : Attention Based Adaptive Spatial-Temporal Graph Convolutional Networks for Stock Forecasting

# Tools
    Python-3.6
    Pytorch-1.6.0

# Dataset:
    ISFD21- 105 stocks from 11 Sectors-Industries
        Date:2001-01-01 to 2021-01-01
        features: Open, High, Low, Close, Adj Close, Volume

    SSFD21- 110 stocks from 11 Sectors
        Date:2001-01-01 to 2021-01-01
        features: Open, High, Low, Close, Adj Close, Volume

# AASTGCN-Run
    # command: python3 ASTGCN_r.py

# Tips
    # Different prediction tasks can be performed by changing the conf file
    # Random initialization of parameters will have a slight impact on the performance of the model
    # the Embdedding dimension of ASTGCN_framework impact the performance of the model,
          for SSFD21, the best emb_dim is 20 and for ISFD21 the best emb_dim is 30 with fine-tune
    if you want to use AASTGCN for other datasets , remember to fine tune the emb_dim in AASTGCN_framework class