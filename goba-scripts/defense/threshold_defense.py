import pandas as pd
import numpy as np

# ---------- 配置 ----------
csv_file = "./data/target_checking_end_positions_dataset_7.csv"
threshold = 1  # 自定义阈值
out_csv = "threshold_labels.csv"

# ---------- 读取数据 ----------
df = pd.read_csv(csv_file)
X = df[["x", "y", "z"]].values
y_true = df["label"].values  # 如果原始数据里有 label

# ---------- 基于第一条数据的阈值分类 ----------
base_point = X[0]  # 第一条数据
distances = np.linalg.norm(X - base_point, axis=1)
y_pred = (distances > threshold).astype(int)

df["cluster_label"] = y_pred

# ---------- 计算准确率 ----------
accuracy = np.mean(y_true == y_pred)
print(f"Threshold Clustering Accuracy: {accuracy:.4f}")

# ---------- 错误率分析 ----------
# 混淆矩阵元素
FP = np.sum((y_true == 0) & (y_pred == 1))  # 0 -> 1
FN = np.sum((y_true == 1) & (y_pred == 0))  # 1 -> 0
TN = np.sum((y_true == 0) & (y_pred == 0))
TP = np.sum((y_true == 1) & (y_pred == 1))

# 概率
p_0_to_1 = FP / (FP + TN) if (FP + TN) > 0 else 0
p_1_to_0 = FN / (FN + TP) if (FN + TP) > 0 else 0


print(f"P(1→0): {p_1_to_0:.4f}")
print(f"P(0→1): {p_0_to_1:.4f}")




# ---------- 保存结果 ----------
df.to_csv(out_csv, index=False)
print(f"结果已保存到 {out_csv}")

