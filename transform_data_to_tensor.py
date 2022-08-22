import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transform
import warnings
from loguru import logger
warnings.filterwarnings("error", category=UserWarning)
from config import random_list, file_batch_size

# 将图片数据集保存为tensor


# 数据集先暂时写入列表 方便打乱顺序
total_list = []

# 第二次预处理
# 将数据集直接储存为tensor的列表
# 方便后续读取
class my_dataset(Dataset):
    def __init__(self, data_list, is_train = True) -> None:
        super().__init__()
        # 储存图片的列表
        self.imgs = []
        self.is_train = is_train
        # 处理数据列表
        for line in data_list:
            # 去掉末尾的\n
            line = line.rstrip()
            # line[-2]为空格 前为路径 后为标签
            self.imgs.append((line[0:-2], int(line[-1])))
        # 训练集的图片处理
        self.trian_transform = transform.Compose(
            [
            # 缩放图片为224*224
            transform.RandomResizedCrop(224),
            # 数据增强
            # 随机水平翻转
            transform.RandomHorizontalFlip(),
            # 随机色彩
            transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3),
            # 随机竖直翻转
            # transform.RandomVerticalFlip(),
            # 转化为 channel * height * width 
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        # 测试集和验证集的图片处理
        self.val_transform = transform.Compose(
            [
            transform.Resize(224),
            transform.ToTensor()
            ]
        )

    def __getitem__(self, index):
        # 打开图片
        file_path, label = self.imgs[index]
        try:
            img = Image.open(file_path).convert('RGB')
            # 缩放填充
            img = self.padding_img(img)
            # 图片处理
            if self.is_train:
                img = self.trian_transform(img)
            else:
                img = self.val_transform(img)
            return img, label
        # 处理损坏的数据集
        except:
            os.remove(file_path)

    def __len__(self):
        return len(self.imgs)

    def padding_img(self, img):
        # 黑色填充层
        padding_backgroud = Image.new('RGB', (224, 224)) 
        # 原图的宽高
        width, height = img.size
        # 缩放比例
        scale = 224. / max(width, height)
        # 缩放
        new_img = img.resize(int(x) for x in [width*scale, height*scale])
        # 缩放后的宽和高
        new_width, new_height =  new_img.size
        # 填充
        padding_backgroud.paste(new_img, ((224 - new_width) // 2, (224 - new_height) // 2))
        return padding_backgroud



# 读取文件名并初步分类到列表中
# 检测到文件夹就前进
# 如果是根目录，检测到文件不返回
# 如果不是根目录，检测到文件就返回
# 相当于一个递归调用
def write_total_list(path, is_root = False, slash = '/'):
    global total_list
    file_names = os.listdir(path)
    for file_name in file_names:
        # 非pt数据集文件夹
        if file_name == 'pt' or file_name == '__pycache__':
            continue
        # 检测到文件夹 进入并继续遍历
        if os.path.isdir(path + slash + file_name):
            write_total_list(path + slash + file_name, False)
        elif not is_root:
            # 初始化最终的文件目录
            file_path = ''
            # 得到最终的文件目录
            file_path = path + slash + file_name
            if '其他垃圾' in path:
                file_path += ' 0\n'
            elif '厨余垃圾' in path:
                file_path += ' 1\n'
            elif '可回收物' in path:
                file_path += ' 2\n'
            else:
                file_path += ' 3\n'
            # 写入文件目录列表
            total_list.append(file_path)
         # 写入每一张列表里面

# 写入各数据集文件
def write_each_file(epoch_num):
    global total_list, file_batch_size, random_list
    # 写入的训练集的列表
    train_list = []
    # 写入的测试集列表
    test_list = []
    # 写入的验证集列表
    val_list = []
    # 最终的tensor列表
    train_tensor = []
    test_tensor = []
    val_tensor = []
    # 打乱列表顺序 
    random.shuffle(total_list)
    #遍历列表并分类
    for one_batch_list in total_list:
        # 7:1比例分出训练集和测试集+验证集
        choice_1 = random.choice(random_list[2:]) < 9
        # 7:3比例分出验证集和测试集
        choice_2 = random.choice(random_list) < 7
        # choice_1 为 True 为训练集 False 为测试集+验证集
        if choice_1:
            train_list.append(one_batch_list)
        else:
            # choice_2 为 True 为验证集 False 为测试集
            if choice_2:
                val_list.append(one_batch_list)
            else:
                test_list.append(one_batch_list)
    # 释放内存
    del total_list
    ('init the file success')
    train_datas = my_dataset(train_list, True)
    test_datas = my_dataset(test_list, False)
    val_datas = my_dataset(val_list, False)
    del train_list, test_list, val_list
    logger.success('load list ok')
        # 进行储存
    # 特别的 对于训练数据, 由于数据量过大
    # 极易造成内存不足
    # 因此采取分批次写入 分批次读取的方式处理文件
    # 采用20 * batch_size的数据大小(本机约1.5s读取时间)
    # 存取训练集的一次batch

    begin = 0
    pace = file_batch_size*20 
    save_time = len(train_datas) // pace
    for save_times in range(len(train_datas) // pace):
        # 最后一个数据集可能很小
        # 因此与倒数第二个合并
        # 最大也不会超过40个file_batch
        if begin + 2*pace <= len(train_datas):
            for index in range(begin, begin+pace):
                train_tensor.append(train_datas[index])
        else:
            for index in range(begin, len(train_datas)):
                train_tensor.append(train_datas[index])
        begin += pace
        torch.save(train_tensor, './pt/train_' + str(save_times) + '.pt')
        logger.success('hava save train_data '+ str(save_times))
        train_tensor = []
    del train_datas
    del train_tensor
    logger.success('train.pt bulit success')
    test_tensor = sava_tensor(test_datas)
    del test_datas
    torch.save(test_tensor, './pt/test_' + str(epoch_num) + '.pt')
    logger.success('test.pt bulit success')
    del test_tensor
    val_tensor = sava_tensor(val_datas)
    del val_datas
    torch.save(val_tensor, './pt/val.pt')
    logger.success('val.pt bulit success')
    del val_tensor
    return save_time

# 保存tensor列表
def sava_tensor(datas):
    data_list = []
    for index in range(len(datas)):
        data_list.append(datas[index])
    return data_list

def reload_file(root_path, epoch_num):
    write_total_list(root_path, True)
    save_time = write_each_file(epoch_num)
    logger.info('train group '+ str(epoch_num), ' save_time: '+ str(save_time))
    return save_time



