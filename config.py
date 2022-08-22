# 存储常量 变量和超参数
import torch


# 训练所需参数
batch_size = 96
num_epochs = 8000
num_workers = 1
lr_init = 0.001
lr_stepsize = 20
weight_decay = 0.01
num_hiddens1 = 1024
num_hiddens2 = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 预处理文件所需参数
file_batch_size = 192
# 总共约八万条数据集
# 数据集采用 70:7:3 的比例划分训练集、验证集和测试集
# 数据集的长度
train_len, test_len, val_len = 0, 0, 0
# 通过random列表确定概率
random_list = [0,1,2,3,4,5,6,7,8,9]
# 文件根目录
root_path = '.'
