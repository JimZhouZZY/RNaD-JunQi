# 基于 Regularized Nash Dynamics 的军棋AI
#### **注意：为了保护我的工作，我仅将部分代码开源，例如神经网络部分我仅上传了一个示例 CNN**
## 成果展示
### 4*2迷你军棋
4*2迷你军棋采用与论文相同的参数与缩小版（32通道）的CNN。训练时的 batch_size=500。

经过约12000个 learner step 得到 loss 如图：

![fig1](https://github.com/JimZhouZZY/RNaD-JunQi/assets/140597003/34bc0d25-99af-4671-be78-88883a00eae9)

与 random player 进行评估，胜率达到约 82% ，evaluator.py 的部分输出如图：

![bs500_itr10000_32papernet_origin_pitrandom](https://github.com/JimZhouZZY/RNaD-JunQi/assets/140597003/b9ab6b34-e0e3-4862-bc1a-667fb9ee491f)

### 10*2缩小版军棋

10*2缩小版军棋包括军棋中除工兵外全部的游戏逻辑，是我向完整版军棋进发的尝试。

10*2迷你军棋采用与论文相同的参数与缩小版（128通道）的CNN。训练时的 batch_size=20。

由于算力有限，我尚未在10*2缩小版军棋进行足够的参数调试测试，且batch_size=20对于军棋的多智能体训练实在太小，因此目前结果不佳。

经过约5000个 learner step 得到 loss 如图：

![Figure_1](https://github.com/JimZhouZZY/RNaD-JunQi/assets/140597003/0a77bc78-9c12-4699-ac72-6b1792fa6eb7)

与 random player 进行评估，胜率达到约 54% ，负率仅约 39% ，evaluator.py 的部分输出如图：
![bs20_itr5000](https://github.com/JimZhouZZY/RNaD-JunQi/assets/140597003/49ea8752-b47c-437d-8fe4-e4f7ac8f262f)

## 环境搭建
#### **注意：仅支持 linux 环境, 开发和训练试用 ubuntu 22.04 20.04**
首先安装带有军棋游戏的 Open Spiel 框架，以及jax, dm-haiku等依赖项
```
pip install open-spiel-junqi==1.4.2 jax dm-haiku
```
克隆此仓库
```
git clone https://github.com/JimZhouZZY/RNaD-JunQi.git
```
进入 game 文件夹，运行 copy.sh 替换文件（此步骤可能需要手动找到 python 的库文件夹并手动替换）
```
cd game
sh copy.sh
```
（此步可省略）检查机器 cpu 数量，根据机器 cpu 数量调整 config.py 中的 _NUM_ACTORS 变量
```
lscpu
```
开始训练
```
sh train.sh
```

