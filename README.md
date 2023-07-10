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

