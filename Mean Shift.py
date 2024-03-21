import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment

# 导入数据
data = pd.read_excel('E:/view_only/mm/m3.xls')
labels_true = pd.read_excel('E:/view_only/mm/Y.xls')
data.columns = range(data.shape[1])
# 对标签进行编码
label_encoder = LabelEncoder()
labels_true = label_encoder.fit_transform(labels_true)

# 创建并拟合Mean Shift
ms = MeanShift().fit(data)
labels_pred = ms.labels_

# 使用匈牙利算法进行类别匹配（解决类别名称可能的不匹配问题）
D = max(labels_pred.max(), labels_true.max()) + 1
w = np.zeros((D, D), dtype=np.int64)
for i in range(labels_pred.size):
    w[labels_pred[i], labels_true[i]] += 1
ind = linear_sum_assignment(w.max() - w)
ind = np.asarray(ind)
ind = np.transpose(ind)
labels_pred = ind[labels_pred, 0]

# 计算评估指标
print('ACC:', metrics.accuracy_score(labels_true, labels_pred))
print('NMI:', metrics.normalized_mutual_info_score(labels_true, labels_pred))
print('Purity:', metrics.cluster.contingency_matrix(labels_true, labels_pred).max(axis=0).sum() / labels_true.shape[0])
print('Fscore:', metrics.f1_score(labels_true, labels_pred, average='macro'))
print('Precision:', metrics.precision_score(labels_true, labels_pred, average='macro'))
print('Recall:', metrics.recall_score(labels_true, labels_pred, average='macro'))
print('AR:', metrics.adjusted_rand_score(labels_true, labels_pred))
