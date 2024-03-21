import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment

# 读取数据集
data = pd.read_excel('E:/view_only/mm/m3.xls')
labels_true = pd.read_excel('E:/view_only/mm/Y.xls').squeeze()
data.columns = range(data.shape[1])
# 将真实标签进行编码
label_encoder = LabelEncoder()
labels_true = label_encoder.fit_transform(labels_true)

# 创建OPTICS模型并进行拟合
optics_model = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics_model.fit(data)

# 获取聚类标签
labels_pred = optics_model.labels_

# 由于OPTICS可能产生噪声点（标记为-1），我们将这些点单独作为一个类别处理
# 这将允许我们在计算性能指标时考虑这些噪声点
max_label = labels_true.max()
labels_pred[labels_pred == -1] = max_label + 1

# 使用匈牙利算法来调整聚类标签与真实标签的对应关系
D = max(labels_pred.max(), labels_true.max()) + 1
w = np.zeros((D, D), dtype=np.int64)
for i in range(labels_pred.size):
    w[labels_pred[i], labels_true[i]] += 1
ind = linear_sum_assignment(w.max() - w)
ind = np.asarray(ind)
ind = np.transpose(ind)
labels_pred = ind[labels_pred, 0]

# 计算聚类性能指标
print('ACC:', metrics.accuracy_score(labels_true, labels_pred))
print('NMI:', metrics.normalized_mutual_info_score(labels_true, labels_pred))
print('Purity:', metrics.cluster.contingency_matrix(labels_true, labels_pred).max(axis=0).sum() / labels_true.shape[0])
print('Fscore:', metrics.f1_score(labels_true, labels_pred, average='macro'))
print('Precision:', metrics.precision_score(labels_true, labels_pred, average='macro'))
print('Recall:', metrics.recall_score(labels_true, labels_pred, average='macro'))
print('AR:', metrics.adjusted_rand_score(labels_true, labels_pred))
