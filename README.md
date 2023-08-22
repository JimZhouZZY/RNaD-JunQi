# 目录
- [简介](#简介)
- [流程](#流程)  
  - [环境搭建](#环境搭建)



## 简介
基于spiel、R-NaD的四国军棋AI,开发中

[Spiel](https://www.deepmind.com/open-source/openspiel)是由DeepMind开发的一个Python库，用于创建和研究各种强化学习游戏。它提供了一个灵活的框架，可以用于定义游戏规则、状态空间、行动空间以及奖励函数等。"Spiel"库还提供了一系列用于评估和比较不同强化学习算法性能的标准游戏，例如围棋、国际象棋、扑克等。spiel的github地址：https://github.com/deepmind/open_spiel

Regularised Nash Dynamics(R-NaD)是一种强化学习算法，被用于西洋陆战棋(Stratego)的AI开发中，取得了人类专家水准。[查看原文](https://arxiv.org/abs/2206.15378)  
Spiel中提供了对该算法的实现

本项目以初学者的视角，基于Spiel，将R-NaD应用于四国军棋中，并记录实现流程。  
__注：游戏部分的api没有doc，只能在open_spiel/spiel.h中找到说明。其他api的用法详见doc__
## 流程  
Spiel对windows的系统支持有限，我们在Ubuntu20.04上进行开发调试，分别用wsl和linux主机进行前期部分流程，训练阶段需配置远程服务器环境

spiel提供了Python和C++的api。使用**python>=3.9**构建项目的流程:  
- 克隆仓库(需要[配置代理](#环境搭建))
- openspiel/python/game下完成junqi.py(游戏类型的配置，棋盘状态和规则的定义，AI所获得的信息等内容)和junqi_test.py
- 到open_spiel/python/games/__init__.py中添加刚才的脚本
- open_spiel/python/tests/pyspiel_test.py中添加游戏短名
- open_spiel/python/CMakeLists.txt中添加junqi_test.py
- 在项目根目录中运行install.sh,会自动安装一些包(需要[配置代理](#环境搭建))
- `python setup.py build`
- `python setup.py install`过程中可能失败，需要更换国内源
- 还没学
- 还没学
- 还没学
- open_spiel/python/examples/下` python example.py --game=junqi`

### 环境搭建
首先将Spiel的仓库克隆到本地，克隆前需**配置网络代理**。参考教程:[wsl使用宿主机代理](https://solidspoon.xyz/2021/02/17/%E9%85%8D%E7%BD%AEWSL2%E4%BD%BF%E7%94%A8Windows%E4%BB%A3%E7%90%86%E4%B8%8A%E7%BD%91/);;;[Ubuntu命令行配置clash(没有测试)](https://www.hengy1.top/article/3dadfa74.html)
```
git clone https://github.com/deepmind/open_spiel
```






