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
## 流程  
Spiel对windows的系统支持有限，我们在Ubuntu20.04上进行开发调试，分别用wsl和linux主机进行前面的部分流程

### 环境搭建
首先将Spiel的仓库克隆到本地
```
git clone https://github.com/deepmind/open_spiel
```



### 简易全明1v1测试
