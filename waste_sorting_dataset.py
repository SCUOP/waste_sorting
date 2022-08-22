from torch.utils.data import Dataset, DataLoader
import random
import torch

#-2为验证模式, -1为测试模式, 0以上为训练模式
class waste_sorting_dataset(Dataset):
    def __init__(self, train_mod_num):
        if train_mod_num == -2:
            self.imgs = torch.load('./pt/val.pt')
        elif train_mod_num == -1:
            self.imgs = torch.load('./pt/test.pt')
        else:
            self.imgs = torch.load('./pt/train_' + str(train_mod_num) + '.pt')
    def __getitem__(self, index):
        img, label = self.imgs[index]
        return img, label
    def __len__(self):
        return len(self.imgs)


# if __name__ == "__main__":
#     # 外部打乱 + 内部的shuffle = True打乱获得随机效果
#     random_train_mod = random.sample(range(0, 13), 13)
#     print(random_train_mod)
#     for train_mod_num in random_train_mod:
#         img, label = run_train_loader(train_mod_num)
#         print(train_mod_num)



        
                