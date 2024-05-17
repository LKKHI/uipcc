#读取data/result.mat文件，将
import scipy.io as sio
# result = sio.loadmat('data/result.mat')
# result = sio.loadmat('data/result_10_0.5_20.mat')
# result = sio.loadmat('data/result_10_0.5_30.mat')
# result = sio.loadmat('data/result_10_0.5_64.mat')

# print('time_len',result['time_len'])
# print('upcc_rmse',result['upcc_rmse'])
# print('upcc_mae',result['upcc_mae'])
# print('ipcc_rmse',result['ipcc_rmse'])
# print('ipcc_mae',result['ipcc_mae'])
# print('uipcc_rmse',result['uipcc_rmse'])
# print('uipcc_mae',result['uipcc_mae'])

for sprasity in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]:
    data_path= f'./data/srpasity/result_10_0.5_20_{sprasity}.mat'
    result = sio.loadmat(data_path)
    print('*'*20)
    print('sprasity', sprasity)
    print('ipcc_rmse', result['ipcc_rmse'])
    print('ipcc_mae', result['ipcc_mae'])
    print('upcc_rmse', result['upcc_rmse'])
    print('upcc_mae', result['upcc_mae'])
    print('uipcc_rmse', result['uipcc_rmse'])
    print('uipcc_mae', result['uipcc_mae'])
