import copy
import numpy as np
import math
from tqdm import tqdm
from model_util import nonzero_user_mean,freeze_random
from evaluation import rmse,mae
# 相似度计算库
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import scipy.io as scio
import pickle
class UPCCModel(object):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = None  # QoS矩阵
        self.u_mean = None  # 每个用户的评分均值
        self.similarity_matrix = None  # 用户相似度矩阵
        self._nan_symbol = -1  # 缺失项标记（数据集中使用-1表示缺失项）

    def _get_similarity_matrix(self, matrix, metric):
        """获取项目相似度矩阵

        Args:
            matrix (): QoS矩阵
            metric (): 相似度计算方法, 可选参数: PCC(皮尔逊相关系数), COS(余弦相似度), ACOS(修正的余弦相似度)

        """
        _m = copy.deepcopy(matrix)
        _m[_m == self._nan_symbol] = 0  # 将缺失项用0代替，以便之后计算
        n_users = matrix.shape[0]
        similarity_matrix = np.zeros((n_users, n_users))

        # 计算相似度矩阵
        for i in tqdm(range(n_users), desc="生成相似度矩阵"):
            row_i = _m[i]
            nonzero_i = np.nonzero(row_i)[0]  # 非0元素所对应的下标
            for j in range(i + 1, n_users):
                row_j = _m[j]
                nonzero_j = np.nonzero(row_j)[0]
                intersect = np.intersect1d(nonzero_i,
                                           nonzero_j)  # 用户i,j共同评分过的项目交集
                if len(intersect) == 0:
                    sim = 0
                else:
                    # 依据指定的相似度计算方法计算用户i,j的相似度
                    try:
                        if metric == 'PCC':
                            # 如果一个项目的评分向量中所有值都相等，则无法计算皮尔逊相关系数
                            if len(set(row_i[intersect])) == 1 or len(
                                    set(row_j[intersect])) == 1:
                                sim = 0
                            else:
                                sim = pearsonr(row_i[intersect],
                                               row_j[intersect])[0]
                        elif metric == 'COS':
                            sim = cosine_similarity(row_i[intersect],
                                                    row_j[intersect])
                        elif metric == 'ACOS':
                            sim = adjusted_cosine_similarity(
                                row_i, row_j, intersect, i, j, self.u_mean)
                    except Exception as e:
                        sim = 0
                similarity_matrix[i][j] = similarity_matrix[j][i] = sim

        return similarity_matrix

    def _get_similarity_users(self, uid, topk=-1):
        """获取相似用户

        Args:
            uid (): 当前用户
            topk (): 相似用户数量, -1表示不限制数量

        Returns:
            依照相似度从大到小排序, 与当前用户最为相似的前topk个相似用户

        """
        assert isinstance(topk, int)
        ordered_sim_uid = (
            -self.similarity_matrix[uid]).argsort()  # 按相似度从大到小排序后, 相似用户对应的索引
        if topk == -1:
            return ordered_sim_uid
        else:
            assert topk > 0
            return ordered_sim_uid[:topk]

    def get_similarity(self, uid_a, uid_b):
        """传入两个uid，获取这两个用户的相似度
        """
        if uid_a == uid_b:
            return float(1)
        if uid_a + 1 > self.matrix.shape[0] or uid_b + 1 > self.matrix.shape[0]:
            return 0
        if self.similarity_matrix is None:
            assert self.matrix is not None, "Please fit first e.g. model.fit()"
            self._get_similarity_matrix(self.matrix)

        return self.similarity_matrix[uid_a][uid_b]

    def fit(self, triad, metric='PCC'):
        """训练模型

        Args:
            triad (): 数据三元组: (uid, iid, rating)
            metric (): 相似度计算方法, 可选参数: PCC(皮尔逊相关系数), COS(余弦相似度), ACOS(修正的余弦相似度)
        """
        # self.matrix = triad_to_matrix(triad, self._nan_symbol)  # 数据三元组转QoS矩阵
        self.matrix = triad
        self.u_mean = nonzero_user_mean(self.matrix,
                                        self._nan_symbol)  # 根据QoS矩阵计算每个用户的评分均值
        self.similarity_matrix = self._get_similarity_matrix(
            self.matrix, metric)  # 根据QoS矩阵获取用户相似矩阵

    # def predict(self, triad, topK=-1):
    #     y_list = []  # 真实评分
    #     y_pred_list = []  # 预测评分
    #     cold_boot_cnt = 0  # 冷启动统计
    #     assert self.u_mean is not None, "Please fit first e.g. model.fit()"
    #
    #     for row in tqdm(triad, desc="Predict... "):
    #         uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
    #         # 冷启动: 新用户因为没有计算过相似用户, 因此无法预测评分
    #         if uid + 1 > len(self.u_mean):
    #             cold_boot_cnt += 1
    #             continue
    #         u_mean = self.u_mean[uid]  # 当前用户评分均值
    #         similarity_users = self._get_similarity_users(uid, topK)
    #         up = 0  # 分子
    #         down = 0  # 分母
    #         # 对于当前用户的每一个相似用户
    #         for sim_uid in similarity_users:
    #             sim_user_rate = self.matrix[sim_uid][iid]  # 相似用户对目标item的评分
    #             similarity = self.get_similarity(uid, sim_uid)
    #             # 如果相似用户对目标item没有评分，或者相似度为负，则不进行计算
    #             if sim_user_rate == self._nan_symbol or similarity <= 0:
    #                 continue
    #             up += similarity * (sim_user_rate - self.u_mean[sim_uid]
    #                                 )  # 相似度 * (相似用户评分 - 相似用户评分均值)
    #             down += similarity
    #
    #         if down != 0:
    #             y_pred = u_mean + up / down
    #         else:
    #             y_pred = u_mean
    #
    #         y_pred_list.append(y_pred)
    #         y_list.append(rate)
    #
    #     print(f"cold boot :{cold_boot_cnt / len(triad) * 100:4f}%")
    #     return y_list, y_pred_list

def predict(removed_matrix,similarity_matrix, topK=10,type ='upcc'):
    num_users = removed_matrix.shape[0]
    num_items = removed_matrix.shape[1]
    if type == 'ipcc': removed_matrix = removed_matrix.T
    pred = np.full((num_users,num_items), -1)
    for i,pcc_for_user_service in enumerate(similarity_matrix):
    #根据pccforuser的值进行从大到小排序
        sorted_indices = np.argsort(pcc_for_user_service)[::-1]
        #取topk个值
        sorted_indices = sorted_indices[:topK]
        if type == 'upcc':
            for sid in range(num_items):
                pccSum = 0
                predValue = 0
                for uid in sorted_indices:
                    userPCCValue = pcc_for_user_service[uid]
                    # if removed_matrix[uid][sid]<0:continue#平均时间内没有调用过该服务
                    pccSum += userPCCValue
                    predValue +=userPCCValue *(removed_matrix[uid][sid]-np.mean(removed_matrix[uid]))
                if pccSum==0:
                    predValue = np.mean(removed_matrix[i])
                else:
                    predValue = predValue/pccSum+np.mean(removed_matrix[i])
                if predValue<=0: predValue = np.mean(removed_matrix[i])
                pred[i][sid] = predValue
        elif type == 'ipcc':
            for uid in range(num_users):
                pccSum = 0
                predValue = 0
                for sid in sorted_indices:
                    servicePCCValue = pcc_for_user_service[sid]
                    # if removed_matrix[uid][sid]<0:continue#平均时间内没有调用过该服务
                    pccSum += servicePCCValue
                    mean = np.mean(removed_matrix[sid])
                    predValue +=servicePCCValue *(removed_matrix[sid][uid]-mean)
                if pccSum==0:
                    predValue = np.mean(removed_matrix[i])
                else:
                    predValue = predValue/pccSum+np.mean(removed_matrix[i])
                if predValue<=0: predValue = np.mean(removed_matrix[i])
                pred[uid][i] = predValue

    return pred

def adjusted_cosine_similarity(x, y, intersect, id_x, id_y, u_mean):
    """修正的余弦相似度

    Returns:

    """
    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    if n < 2:
        raise ValueError('x and y must have length at least 2.')
    if len(intersect) < 2:
        raise ValueError('there must be at least two non-zero entries')

    x = np.asarray(x)
    y = np.asarray(y)
    nonzero_x = np.nonzero(x)[0]
    nonzero_y = np.nonzero(y)[0]

    multiply_sum = sum(
        (x[i] - u_mean[id_x]) * (y[i] - u_mean[id_y]) for i in intersect)
    pow_sum_x = sum(math.pow(x[i] - u_mean[id_x], 2) for i in nonzero_x)
    pow_sum_y = sum(math.pow(y[i] - u_mean[id_y], 2) for i in nonzero_y)

    return multiply_sum / math.sqrt(pow_sum_x * pow_sum_y)

def fit_and_sum(args):
    i, (t, upcc) = args
    upcc.fit(t, metric='PCC')
    return i, upcc.similarity_matrix, t

def fit_and_sum_T(args):
    i, (t, upcc) = args
    upcc.fit(t.T, metric='PCC')
    return i, upcc.similarity_matrix, t

def train_upcc(train_qos,topK=10,time_len=64):
    #进行upcc模型训练
    #创建64个UPCCModel()对象
    upcc_list = [UPCCModel() for _ in range(time_len-1)]
    similarity_matrix_sum = np.random.normal(size=(test_qos.shape[0],test_qos.shape[0]))
    # matrix_sum = np.random.normal(size=(test_qos.shape[0],test_qos.shape[1]))
    matrix_sum = np.zeros_like(test_qos)
    #获得每个时间的相似度矩阵
    # for i,t in enumerate (train_qos):
    #     upcc_list[i].fit(t, metric='PCC')
    #     similarity_matrix_sum += upcc_list[i].similarity_matrix
    #     matrix_sum += t


    # 假设 upcc_list 是一个包含 UPCC 对象的列表
    # train_qos 是一个包含训练数据的列表
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(fit_and_sum, enumerate(zip(train_qos, upcc_list))))
    sim=[]
    # 将结果合并
    for i, sim_matrix, t in results:
        sim.append(sim_matrix)
        similarity_matrix_sum += sim_matrix
        matrix_sum += t

    matrix_sum = matrix_sum/ (time_len-1)
    similarity_matrix_sum = similarity_matrix_sum/(time_len-1)
    #保存相似度矩阵
    #根据相似度矩阵进行预测
    pred = predict(matrix_sum,similarity_matrix_sum, topK)
    return pred,sim

def train_ipcc(train_qos,topK=10,time_len=64):
    #进行upcc模型训练
    #创建64个UPCCModel()对象
    upcc_list = [UPCCModel() for _ in range(time_len-1)]
    similarity_matrix_sum = np.random.normal(size=(test_qos.shape[1],test_qos.shape[1]))
    matrix_sum = np.random.normal(size=(test_qos.shape[0],test_qos.shape[1]))
    # matrix_sum = np.zeros_like(test_qos)

    #单线程计算(通俗易懂版本)
    #获得每个时间的相似度矩阵
    # for i,t in enumerate (train_qos):
    #     upcc_list[i].fit(t.T, metric='PCC')
    #     similarity_matrix_sum += upcc_list[i].similarity_matrix
    #     matrix_sum += t.T
    # 假设 upcc_list 是一个包含 UPCC 对象的列表
    # train_qos 是一个包含训练数据的列表

    #多线程计算
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(fit_and_sum_T, enumerate(zip(train_qos, upcc_list))))
    sim = []
    # 将结果合并
    for i, sim_matrix, t in results:
        sim.append(sim_matrix)
        similarity_matrix_sum += sim_matrix
        matrix_sum += t
    matrix_sum = matrix_sum/ (time_len-1)
    similarity_matrix_sum = similarity_matrix_sum/(time_len-1)
    #根据相似度矩阵进行预测
    pred = predict(matrix_sum,similarity_matrix_sum, topK,type ='ipcc')
    return pred,sim
def print_result(pred,test_qos):
    rm = rmse(test_qos,pred)
    ma = mae(test_qos,pred)
    print('rmse:',rm)
    print('mae:',ma)
    return rm,ma
if __name__ == "__main__":
    freeze_random()  # 冻结随机数 保证结果一致

    #参数
    topK = 10
    alph = 0.5
    time_len=20

    #读取('../../data/rt.npy')
    data = np.load('./data/rt.npy', allow_pickle=True)
    train_qos = data[:time_len-1,:,:]
    test_qos = data[time_len-1,:,:]
    print('upcc模型训练')
    #进行upcc模型训练
    pred_upcc,user_similarity_matrix = train_upcc(train_qos,topK,time_len)
    print('upcc预测:')
    upcc_rmse,upcc_mae = print_result(pred_upcc,test_qos)
    #保存list
    with open('./data/user_similarity_matrix_{topK}_{alph}_{time_len}.pkl', 'wb') as f:
        pickle.dump(user_similarity_matrix, f)
    #进行ipcc训练
    print('ipcc模型训练')
    pred_ipcc,service_similarity_matrix = train_ipcc(train_qos,topK,time_len)
    print('ipcc预测:')
    ipcc_rmse,ipcc_mae = print_result(pred_ipcc,test_qos)
    with open('./data/service_similarity_matrix_{topK}_{alph}_{time_len}.pkl', 'wb') as f:
        pickle.dump(service_similarity_matrix, f)
    #进行uipcc训练
    print('uipcc模型训练')
    pred_uipcc = alph * pred_upcc+ (1-alph)*pred_ipcc
    print('uipcc预测:')
    uipcc_rmse,uipcc_mae = print_result(pred_uipcc,test_qos)
    #将上面数据保存到mat中
    scio.savemat(f'./data/result_{topK}_{alph}_{time_len}.mat',{
        'topK':topK,
        'alph':alph,
        'time_len':time_len,
        'upcc_rmse':upcc_rmse,
        'upcc_mae':upcc_mae,
        'ipcc_rmse':ipcc_rmse,
        'ipcc_mae':ipcc_mae,
        'uipcc_rmse':uipcc_rmse,
        'uipcc_mae':uipcc_mae}
                 )

