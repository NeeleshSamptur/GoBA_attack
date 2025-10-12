import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mode

# ---------- 配置 ----------
csv_file = "./data/target_checking_end_positions_dataset_7.csv"
n_clusters = 2
random_state = 7

# ---------- 读取数据 ----------
df = pd.read_csv(csv_file)
X = df[["x", "y", "z"]].values
y_true = df["label"].values  # 0=正样本, 1=负样本

# ---------- KMeans 聚类 ----------
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
y_pred = kmeans.fit_predict(X)

# ---------- 聚类结果映射到 label ----------
def map_clusters_to_labels(y_true, y_pred):
    mapped = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = (y_pred == cluster)
        if np.any(mask):
            majority_label = mode(y_true[mask], keepdims=True).mode[0]
            mapped[mask] = majority_label
    return mapped

y_mapped = map_clusters_to_labels(y_true, y_pred)
df["cluster_label"] = y_mapped

# ---------- 计算准确率 ----------
accuracy = np.mean(y_true == y_mapped)
print(f"Clustering Accuracy: {accuracy:.4f}")

# ---------- 计算混淆矩阵元素 ----------
TP = np.sum((y_true == 0) & (y_mapped == 0))  # 正样本预测为正
FN = np.sum((y_true == 0) & (y_mapped == 1))  # 正样本预测为负
FP = np.sum((y_true == 1) & (y_mapped == 0))  # 负样本预测为正
TN = np.sum((y_true == 1) & (y_mapped == 1))  # 负样本预测为负

# ---------- 计算 FPR / FNR ----------
FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # 负样本误判为正样本概率
FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # 正样本误判为负样本概率

print(f"FPR (1→0): {FPR:.4f}")
print(f"FNR (0→1): {FNR:.4f}")

# ---------- 保存结果 ----------
df.to_csv("kmeans_cluster_labels.csv", index=False)
print("聚类结果已保存到 kmeans_cluster_labels.csv")
