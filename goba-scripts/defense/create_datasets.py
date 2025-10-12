import h5py
import numpy as np
import pandas as pd

file_path1 = "./LIBERO/libero/datasets_orig/libero_object_no_noops/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
file_path2 = "../new3/trigger_basket/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"

inject_rate = 0.18   # 最终 dataset 中 file2 占比
seed = 7
out_csv = "./data/trigger_basket_end_positions_dataset_7.csv"

# ---------- helper ----------
def extract_ends(h5_path, ee_key="ee_pos"):
    ends = []
    names = []
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            return np.empty((0,3)), []
        data_grp = f["data"]
        demo_keys = sorted(data_grp.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k and k.split("_")[-1].isdigit() else k)
        for k in demo_keys:
            try:
                traj = np.array(data_grp[k]["obs"][ee_key])
                if traj.size == 0 or traj.shape[1] < 3:
                    continue
                ends.append(traj[-1,:3])
                names.append(k)
            except:
                continue
    if len(ends) == 0:
        return np.empty((0,3)), []
    return np.vstack(ends), names

# ---------- extract ----------
ends1, names1 = extract_ends(file_path1)
ends2, names2 = extract_ends(file_path2)

n1 = len(ends1)
n2 = len(ends2)
print(f"file1: {n1} demo, file2: {n2} demo")

# ---------- 计算抽样数量 ----------
# inject_rate = n2_sample / (n1 + n2_sample) -> n2_sample = inject_rate * n1 / (1 - inject_rate)
n_sample = int(round(inject_rate * n1 / (1 - inject_rate)))
n_sample = min(n_sample, n2)  # 不超过 file2 的 demo 数量
print(f"从 file2 抽取 {n_sample} 个 demo")

# 随机抽样
rng = np.random.default_rng(seed)
if n_sample > 0:
    idx = rng.choice(n2, size=n_sample, replace=False)
    sampled2 = ends2[idx]
    sampled2_names = [names2[i] for i in idx]
else:
    sampled2 = np.empty((0,3))
    sampled2_names = []

# ---------- 合并 ----------
X = np.vstack([ends1, sampled2])
labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n_sample, dtype=int)])
sources = ["file1"]*n1 + ["file2"]*n_sample
demo_ids = names1 + sampled2_names

# 构造 DataFrame
df = pd.DataFrame(X, columns=["x","y","z"])
df["label"] = labels
df["source"] = sources
df["demo_id"] = demo_ids

# ---------- 随机打乱 ----------
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# 保存 CSV
df.to_csv(out_csv, index=False)

print(f"保存 CSV: {out_csv}")
print(f"最终样本数: {len(df)} (file1: {n1}, file2: {n_sample})")