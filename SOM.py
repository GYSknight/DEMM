import pandas as pd
from minisom import MiniSom
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# 读取数据集
data = pd.read_excel('E:/view_only/TCGA_EJ/ej3.xls').values
labels_true = pd.read_excel('E:/view_only/TCGA_EJ/Y.xls').squeeze()
# data.columns = range(data.shape[1])
# 将真实标签进行编码
label_encoder = LabelEncoder()
labels_true = label_encoder.fit_transform(labels_true)

# SOM模型初始化
som = MiniSom(x=10, y=10, input_len=data.shape[1], sigma=1.0, learning_rate=0.5)

# 训练SOM
som.train_random(data, 100)

# 将数据映射到SOM网格中
winning_nodes = [som.winner(d) for d in data]

# 使用K-means进行后处理以获取标签
from sklearn.cluster import KMeans

cluster_labels = KMeans(n_clusters=len(set(labels_true))).fit_predict(winning_nodes)

# 计算聚类性能指标
print('ACC:', metrics.accuracy_score(labels_true, cluster_labels))
print('NMI:', metrics.normalized_mutual_info_score(labels_true, cluster_labels))
print('Purity:', metrics.cluster.contingency_matrix(labels_true, cluster_labels).max(axis=0).sum() / labels_true.shape[0])
print('Fscore:', metrics.f1_score(labels_true, cluster_labels, average='macro'))
print('Precision:', metrics.precision_score(labels_true, cluster_labels, average='macro'))
print('Recall:', metrics.recall_score(labels_true, cluster_labels, average='macro'))
print('AR:', metrics.adjusted_rand_score(labels_true, cluster_labels))
