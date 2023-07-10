# 多模态情感分析

## 简介

这是DASE当代人工智能2023学年课程，第五次实验的代码仓库。实现了一个多模态融合模型用于处理一个图文数据-情感标签三分类任务。本项目基于paddle框架实现，多模态模型使用了ResNet和一维CNN，实验中可通过命令行指定ResNet层数。

## 安装环境

```
matplotlib~=3.7.1
numpy~=1.24.3
paddlepaddle_gpu~=2.5.0
Pillow~=9.4.0
```

```
pip install -r requirements.txt
```

## 项目结构

```
│  README.md
│  requirements.txt
│
├─predictions # 存放生成的预测结果，格式和train.txt相同
│      predict.txt 
│
├─raw_data # 源数据
│  │  test_without_label.txt
│  │  train.txt
│  │
│  └─data # data子目录下是具体的每行数据对应的文本txt文件和图片jpg文件
├─saved_models # 保存的训练后的模型
└─utils
    │  ablation_eval.py # 消融实验组件
    │  data_load.py	# 从新生成的tmp_data中加载数据，生成dataloader
    │  data_process.py # 处理源数据生成易于操作的数据（位于tmp_data）
    │  model.py # 构建的模型
    │  predict.py # 使用保存的训练后的模型进行预测
    │  run.py # 训练模型，并进行验证和预测
    │
    └─tmp_data # 由data_process.py生成的便于处理的临时数据
            all_data.txt
            all_test_data.txt
            dict.txt
            test_data.txt
            train_data.txt
            val_data.txt
```

## 执行代码

1、进入utils目录，并执行run.py，以默认参数训练并在测试集上预测一次，生成预测结果txt文件

```
cd utils
python3 run.py
```

2、进入predictions目录，查看预测结果predict.txt

```
cd predictions
ls
```

3、可通过命令行设置的其他超参数和训练要求，以下命令代表着执行20个epochs的训练，初始学习率为0.0002，resnet层数类型为resnet18，启动学习率衰减，结束训练时画图描述指标变化情况

```
python3 run.py –epoch 20 –learning_rate 0.0002 –res_level 18 –lr_decay –draw
```

4、使用保存的训练后模型在测试集上预测（训练后的模型保存在saved_models目录下，便于上传仓库中把model进行了压缩，按步骤运行代码后将在saved_models目录下生成未压缩的完整模型）

```
cd utils
python3 predict.py
```

5、对模型进行消融实验

执行utils目录下的ablation_eval.py，默认时同时加载图片和文本数据

```
cd utils
python3 ablation_eval.py
```

只加载图片数据

```
python3 ablation_eval.py --only_info image
```

只加载文本数据

```
python3 ablation_eval.py --only_info text
```

## 验证集结果

| 加载验证集中的不同数据                             | Accuracy |
| -------------------------------------------------- | -------- |
| Full（和训练中一样，使用验证集中所有的图像和文本） | 0.4708   |
| Only Text（只有文本，图像替换为空白数据）          | 0.6167   |
| Only Image（只有图像，文本替换为空白数据）         | 0.4583   |

## 参考

项目代码参考的教程和库：

[ResNet等model的实现](https://github.com/pytorch/vision/tree/main/torchvision/models)

[Paddle上手](https://github.com/PaddlePaddle/Paddle)

[多模态模型相关基础知识](https://zhuanlan.zhihu.com/p/475734302)
