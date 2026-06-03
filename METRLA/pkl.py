import pickle

# 设定pkl文件的路径
pkl_file_path = '/root/autodl-tmp/test1/METRLA/adj_mat_with_newnodes.pkl'
pkl_file_path = '/root/autodl-tmp/test1/METRLA/adj_mat_new.pkl'
# 使用 pickle 加载文件
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# 查看文件内容
print(data)