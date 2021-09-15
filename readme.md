#  余弦退火从启动学习率机制 
【导语】主要介绍 ** 在pytorch 中实现了余弦退火重启动学习率机制，支持 warmup 和 resume 训练。并且支持自定义下降函数，实现多种重启动机制。

代码： https://github.com/Huangdebo/CAWB

## 1. 多 step 重启动
![设定 cawb_steps 之后，便可实现多步长余弦退火重启动学习率机制](https://img-blog.csdnimg.cn/bd85817d24bb400b92c53fe14ead9c1e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd29uZGVyZnVsX2hkYg==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
设定 cawb_steps 之后，便可实现多步长余弦退火重启动学习率机制。每次重启动时，开始学习率会乘上一个比例因子 step_scale。

## 2. 正常余弦退火机制
![如果 cawb_steps 为 [], 则会实现正常的余弦退火机制](https://img-blog.csdnimg.cn/50ad359e69f2459cbbe3268a1be821e4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd29uZGVyZnVsX2hkYg==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
如果 cawb_steps 为 [], 则会实现正常的余弦退火机制，在整个 epochs 中按设定的 lf 机制一直下降

## 3. warmup
![设定 warmup_epoch 之后便可实现学习率的 warmup 机制](https://img-blog.csdnimg.cn/e2e1ad5fe3d0496aa4d674018c1191d8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd29uZGVyZnVsX2hkYg==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
设定 warmup_epoch 之后便可实现学习率的 warmup 机制。warmup_epoch 结束后则按设定的 cawb_steps 实现重启动退火机制。

## 4. resume
![设定 last_epoch 便可实现 resume 训练](https://img-blog.csdnimg.cn/a17f69421aa74c44a1acd3ad737094f9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd29uZGVyZnVsX2hkYg==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
设定 last_epoch 便可实现 resume 训练，接上之前中断的训练中的学习率。

## 5. 自定义下降函数
![自定义下降函数](https://img-blog.csdnimg.cn/3d16857453d44cabb93294edf7ce1ede.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd29uZGVyZnVsX2hkYg==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
可通过自定义下降函数，实现多种重启动机制

```python
# lf = lambda x, y=opt.epochs: (((1 + math.cos(x * math.pi / y)) / 2) ** 1.0) * 0.9 + 0.1  
lf = lambda x, y=opt.epochs: (1.0 - (x / y)) * 0.9 + 0.1 
scheduler = CosineAnnealingWarmbootingLR(optimizer, epochs=opt.epochs, steps=opt.cawb_steps, 
                                             lf=lf, batchs=len(data), warmup_epoch=0)
```





