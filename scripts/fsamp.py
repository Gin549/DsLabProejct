def fsamp(nsamp,df):
    size = df.shape
    step = int(100/nsamp)
    index = [i for i in range(0,size[0],step)]
    sampledDf = df.iloc[index,:].reset_index(drop=True)
    return sampledDf

