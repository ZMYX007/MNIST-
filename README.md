# MNIST-
MNIST 手写数字图像分类可视化项目
MNIST 多学习率对比 CNN 分类项目

项目简介
本项目基于 PyTorch 实现手写数字 MNIST 十分类任务，做了工程化拓展，可用于简历深度学习实战项目。

核心功能：
超参数解耦：独立配置文件统一管理路径、训练参数，无需修改主程序即可调整实验
多学习率对照实验：批量循环多组学习率训练，每组生成独立 TensorBoard 日志，可视化对比损失、准确率曲线完成超参数调优
完整模型 Checkpoint 存取：保存模型权重、优化器状态、实验参数，支持模型重载用于推理 / 二次训练
混淆矩阵可视化：结合 scikit-learn 与 matplotlib 生成分类混淆图，写入 TensorBoard 直观分析类别错分情况
自适应设备训练：自动识别 GPU (CUDA)/CPU 环境，兼容不同硬件设备
TensorBoard 全维度可视化：训练 / 验证标量曲线、样本图像、网络计算图

技术栈
Python 3.x | PyTorch | torchvision | TensorBoard | scikit-learn | Matplotlib | NumPy

项目目录结构
plaintext
mnist_tb_project/
├── config.py           # 全局超参数&路径配置文件
├── train_compare.py   # 主训练脚本，多lr实验、模型保存、混淆矩阵可视化
├── requirements.txt    # 项目依赖清单
├── README.md           # 项目说明文档（本文档）
├── data/               # 自动生成，MNIST数据集存储目录
├── logs/               # 自动生成，TensorBoard日志根目录
│   └── compare_lr/    # 各组学习率独立日志文件夹
└── saved_models/       # 自动生成，训练完成模型checkpoint存储目录

运行指引
1. 启动批量多学习率实验
直接运行主脚本，程序将自动循环执行config.py中LR_LIST定义的全部学习率实验：
powershell
python train_compare.py
训练过程会输出每一轮测试集准确率，完成后自动保存对应学习率的模型文件至saved_models。
2. TensorBoard 可视化查看实验结果
训练全部结束后，在终端执行以下命令启动可视化面板：
powershell
tensorboard --logdir=./logs/compare_lr
复制输出链接（如http://localhost:6006）在浏览器打开：
Scalars：对比不同学习率下训练损失、测试损失、测试准确率变化曲线
Images：查看训练集手写数字样本
Graphs：查看 CNN 网络计算结构图
Figures：查看混淆矩阵图像，分析数字分类错分情况
3. 加载已保存模型（演示）
取消代码末尾load_demo()注释，重新运行脚本，可演示读取本地保存的模型权重，打印对应实验参数。

升级亮点
配置解耦：独立config.py管理所有超参数、路径，无需修改主代码即可调整实验，工程化规范。
多超参数对照实验：循环运行多组学习率，每组独立 TensorBoard 日志目录，在面板中一键对比 Loss/Acc 曲线，完成超参数调优实验。
完整模型存取：使用torch.save保存 checkpoint（含模型权重、优化器状态、超参数），提供加载函数，支持断点续训 / 推理部署。
混淆矩阵可视化：集成scikit-learn混淆矩阵 + matplotlib绘图，将分类结果混淆图写入 TensorBoard，直观分析类别错分情况。
设备自适应：自动检测 CUDA/GPU，兼容 CPU、GPU 运行。
