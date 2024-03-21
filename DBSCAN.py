import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np

# 自定义的 purity_score 函数
def purity_score(y_true, y_pred):
    # 计算纯度
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# 读取数据
data = pd.read_excel('E:/view_only/mm/m3.xls')
labels_true = pd.read_excel('E:/view_only/mm/Y.xls').values.ravel()
# 转换列名为整数类型
data.columns = range(data.shape[1])
# 标准化数据
data = StandardScaler().fit_transform(data)

# DBSCAN 聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(data)

# 计算聚类指标
acc = metrics.accuracy_score(labels_true, clusters)
nmi = metrics.normalized_mutual_info_score(labels_true, clusters)
purity = purity_score(labels_true, clusters) # 使用自定义的函数计算纯度
fscore = metrics.f1_score(labels_true, clusters, average='weighted')
precision = metrics.precision_score(labels_true, clusters, average='weighted')
recall = metrics.recall_score(labels_true, clusters, average='weighted')
ar = metrics.adjusted_rand_score(labels_true, clusters)

# 输出结果
print(f'ACC: {acc}')
print(f'NMI: {nmi}')
print(f'Purity: {purity}')
print(f'Fscore: {fscore}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'AR: {ar}')
