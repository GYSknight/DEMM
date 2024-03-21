import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy import stats

# 加载特征数据
data = pd.read_excel('E:/view_only/chaos/CHAOS1.xls')

# 加载类标签数据
labels = pd.read_excel('E:/view_only/chaos/Y.xls')

# 确保数据是np.array格式
X = np.array(data)
y_true = np.array(labels).flatten()

# 执行K-Means聚类
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
y_pred = kmeans.labels_

# 定义purity的计算方法
def purity_score(y_true, y_pred):
    # 计算混淆矩阵
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # 返回purity值
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# 计算ACC
acc = metrics.accuracy_score(y_true, y_pred)
# 计算NMI
nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
# 计算Purity
purity = purity_score(y_true, y_pred)
# 计算Fscore
fscore = metrics.f1_score(y_true, y_pred, average='weighted')
# 计算Precision
precision = metrics.precision_score(y_true, y_pred, average='weighted')
# 计算Recall
recall = metrics.recall_score(y_true, y_pred, average='weighted')
# 计算AR
ar = metrics.adjusted_rand_score(y_true, y_pred)

print('ACC:', acc)
print('NMI:', nmi)
print('Purity:', purity)
print('Fscore:', fscore)
print('Precision:', precision)
print('Recall:', recall)
print('AR:', ar)
