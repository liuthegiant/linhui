import pickle
import pandas as pd
import numpy as np

# === 读取 adj_mx.pkl ===
with open("adj_mat_new.pkl", "rb") as f:
    sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

print("传感器数量:", len(sensor_ids))
print("adjacency matrix 形状:", adj_mx.shape)
print("adjacency matrix 形状:", adj_mx)
# === 保存 adjacency matrix 为 CSV ===
pd.DataFrame(adj_mx, index=sensor_ids, columns=sensor_ids).to_csv("adjacency_matrix.csv")

print("邻接矩阵已保存为 adjacency_matrix.csv")
