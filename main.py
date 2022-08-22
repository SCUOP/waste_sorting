import torch
from waste_sorting_dataset import waste_sorting_dataset
from torch.utils.data import DataLoader
from torchvision import models
from train import train
from torch import optim, nn
from loguru import logger
from config import batch_size, num_workers, num_epochs, lr_init, \
    lr_stepsize, weight_decay, device, num_hiddens1, num_hiddens2
logger.add('waste_sorting.log')

if __name__ == '__main__':
    # 采用resnet34网络
    # 核心代码！！！
    net = models.resnet34(weights = models.ResNet34_Weights.IMAGENET1K_V1)
    net.fc = nn.Sequential(
        nn.Linear(512, 4)
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(num_hiddens1, num_hiddens2),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(num_hiddens2, 4)
        # nn.Linear(256, 4)
    )
    # net.load_state_dict(torch.load('./model_5.pt'))
    # 初始化数据
    # 更新train_iter
    def update_train_iter(train_mod_num):
        logger.info('update train_iter ' + str(train_mod_num))
        train_dataset = waste_sorting_dataset(train_mod_num)
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = num_workers)                     
        return train_loader
    # 初始化验证集的迭代器
    def init_val_iter():
        val_data = waste_sorting_dataset(-2)
        val_iter = DataLoader(dataset = val_data, batch_size = batch_size, pin_memory = True, num_workers = num_workers)
        logger.info('nums of val_data: '+str(len(val_data)))
        return val_iter
    # 初始化测试集的迭代器
    def init_test_iter():
        test_data = waste_sorting_dataset(-1)
        test_iter = DataLoader(dataset = test_data, batch_size = batch_size, pin_memory = True, num_workers = num_workers)
        logger.info('nums of test_iter: '+ str(len(test_data)))
        return test_iter
    # Adam优化
    optimizer = optim.Adam(net.parameters(), lr = lr_init, weight_decay = weight_decay)
    # 调整学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = lr_stepsize, gamma = 0.1)

    # 训练
    train(net, update_train_iter, init_val_iter, optimizer, scheduler, device, num_epochs)
    
    # 测试

