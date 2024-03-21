import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# 读取数据集
data = pd.read_excel('E:/view_only/mm/m3.xls')
labels_true = pd.read_excel('E:/view_only/mm/Y.xls').squeeze()
data.columns = range(data.shape[1])
# 将真实标签进行编码
label_encoder = LabelEncoder()
labels_true = label_encoder.fit_transform(labels_true)

# 创建CURE模型并进行拟合
cure_model = AgglomerativeClustering(n_clusters=7, linkage='single')
cure_model.fit(data)

# 获取聚类标签
labels_pred = cure_model.labels_

# 计算聚类性能指标
print('ACC:', metrics.accuracy_score(labels_true, labels_pred))
print('NMI:', metrics.normalized_mutual_info_score(labels_true, labels_pred))
print('Purity:', metrics.cluster.contingency_matrix(labels_true, labels_pred).max(axis=0).sum() / labels_true.shape[0])
print('Fscore:', metrics.f1_score(labels_true, labels_pred, average='macro'))
print('Precision:', metrics.precision_score(labels_true, labels_pred, average='macro'))
print('Recall:', metrics.recall_score(labels_true, labels_pred, average='macro'))
print('AR:', metrics.adjusted_rand_score(labels_true, labels_pred))
