import numpy as np
import pandas as pd


def from_ori2matrix(df,type,sprasity=0.1,time_len=63):
    time_group=df.groupby('t')
    result_list = []
    for t in range(time_len):
        datas = time_group.get_group(t).values
        # print(data)
        a = np.full((142, 4500), -1.0)
        if t!=time_len-1:
            #根据denstiy去除datas中的数据
            # 根据密度计算要删除的行数
            num_rows_to_remove = int(len(datas) * sprasity)
            # 生成要删除的随机索引
            indices_to_remove = np.random.choice(len(datas), size=num_rows_to_remove, replace=False)
            # 从datas中删除行
            datas = np.delete(datas, indices_to_remove, axis=0)
        for data in datas:
            data.astype(float)
            u = data[0]
            s = data[1]
            a[int(u)][int(s)] = data[3]  # 响应时间
        # print(a)
        a = a[:, :, None]
        result_list.append(a)
    result = np.concatenate(result_list, axis=2)
    #调整维度
    result=np.transpose(result,(2,0,1))
    if type=='rt':
        result=np.where(result==0,-1,result)
        result = np.where(result >= 20, -1, result)
    else:
        result=np.where(result==0,-1,result)

    return result

def get_data_npy(dataset,sprasity=0.3,time_len=63):
    dataset_dir='./dataset'
    dataset_dir+=f'/{dataset}data.txt'#rt or tp
    time_df=pd.read_csv(dataset_dir,names=['u','s','t','tp'], header=None, sep=" ")
    matrix=from_ori2matrix(time_df,dataset,sprasity,time_len)
    return matrix

