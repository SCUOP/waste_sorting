# waste_sorting
machine learning and image identification for waste_sorting 


大一做的卷积神经网络的深度学习，代码是大概七月15日左右写完的，后续经过少许修改，先暂时做个总结。

任务是具体是实现四种垃圾的分类，采用torchvison中的resnet34的模型和models.ResNet34_Weights.IMAGENET1K_V1的初始化参数。主要技术方面在于数据预处理以便于提升速度和在我的RTX3060的笔记本上能够进行训练。

水平有限，只是写个玩玩，只有数据预处理的代码感觉还行（全部我自己写的部分），其他都或多或少参考了一下github和资料书的内容，代码的注释应该还算比较详细了，按道理非常好懂

具体写法请参考网页 https://www.scuop.top/1151.html
