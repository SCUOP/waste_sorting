from torch import gather
from loguru import logger
import torch
import time
import random
from draw import semilogy

# 开始训练
# 传入函数而非对象 减少内存的占用 用的时候再实例化
def train(net, update_train_iter, init_val_iter, optimizer, scheduler, device, num_epochs):
    from config import root_path
    from transform_data_to_tensor import reload_file
    # 转移网络至gpu
    net = net.to(device)
    logger.critical('training on: '+ device.__str__())
    # 分类问题采用交叉熵计算误差
    loss = torch.nn.CrossEntropyLoss()
    save_time = 18
    train_ac = []
    val_ac = []
    for epoch in range(num_epochs):
        # 20为一组重载数据
        # if epoch % 20 == 0:
        if epoch != 0 and epoch % 20 == 0:
            epoch_num = int(epoch / 20)
            logger.info('making datas for training group ' + str(epoch_num))
            save_time = reload_file(root_path, epoch_num)
        train_loss_sum, train_accuracy_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        random_train_mod = random.sample(range(0, save_time), save_time)
        # 训练 
        for train_mod_num in random_train_mod:
            iter = update_train_iter(train_mod_num)
            for img, label in iter:
                # 转移矩阵至cuda
                img = img.to(device)
                label = label.to(device)
                # 计算值
                label_hat = net(img)
                # 计算误差
                l = loss(label_hat, label)
                # 梯度清零
                optimizer.zero_grad()
                # 求梯度
                l.backward()
                # 优化
                optimizer.step()
                # 计算训练误差总值
                train_loss_sum += l.cpu().item()
                # 计算训练准确度
                train_accuracy_sum += (label_hat.argmax(dim = 1) == label).sum().cpu().item()
                n += label.shape[0]
                batch_count += 1
        train_ac.append(train_accuracy_sum/batch_count)
        #保存模型
        net.eval()
        torch.save(net.state_dict(), './pt/model_' + str(epoch) + '.pt')
        net.train()
        scheduler.step()
        # 每5轮进行一次验证
        # if (epoch + 1) % 5 == 0:
        #     # 验证
        #     iter = init_val_iter()
        #     test_accuracy = evaluate_accuracy(iter, net)
        #     logger.critical('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec'
        #         % (epoch + 1, train_loss_sum / batch_count, train_accuracy_sum / n, test_accuracy, time.time() - start))      
        # else:
        #     logger.critical('epoch %d, loss %4f, train acc %.3f'
        #     % (epoch + 1, train_loss_sum / batch_count, train_accuracy_sum / n))

        # 每一轮验证
        iter = init_val_iter()
        test_accuracy = evaluate_accuracy(iter, net)
        val_ac.append(test_accuracy)
        logger.critical('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec'
            % (epoch + 1, train_loss_sum / batch_count, train_accuracy_sum / n, test_accuracy, time.time() - start))
        # semilogy(range(1, epoch+2), train_ac, 'epochs', 'accuracy', range(1, epoch+2), val_ac, ['train', 'valid'])
        

# 计算准确率
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n
        