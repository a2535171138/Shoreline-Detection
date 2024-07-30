# AI-Driven Shoreline Mapping for Coastal Monitoring
这个项目使用深度学习和图像处理方法自动绘制海岸线，包括完整的算法工程和易于使用的用户界面

## Algorithm

### 数据介绍
my_project/  
│  
├── Argus goldcoast/  
│   └── ...  
├── Argus goldcoast shadow/  
│   └── ...  
├── Argus narrabeen/  
│   └── ...  
├── train_set.csv  
├── test_set.csv 
│ 
├── Algorithm  
│   └── ...  
└── ...  
csv：保存用于训练或测试的所有数据，path列是每张图像的路径，label列是已标注的海岸线像素点集，其他列为当前图像的其他特征类别，不会影响训练。  
图像数据：大小不等的RGB海岸图像  
说明：该项目数据集非公开数据，来自[Water Research Laboratory (UNSW Sydney)](https://www.unsw.edu.au/research/wrl)
<img src="sample.png" alt="Dataset Samples" width="500"/>

### 数据预处理
将不同场景的csv数据合并后并输出成可训练的csv文件
```bash
python Algorithm/DataProcessing/process_dataframes.py --csv_files coastsnap_segment_clean.csv argus_goldcoast_segment.csv segment_narraV2.csv plan.csv --folders 'CoastSnap' 'Argus goldcoast' 'Argus narrabeen' --output_csv data_set.csv
```
数据类别平衡，使用--column来specify需要平衡的特征，该特征下所有类型的数据数量都将相同
```bash
python Algorithm/DataProcessing/balance_dataset.py --input_csv data_set.csv --output_csv balanced_data_set.csv --column site
```
难例加权，使用--column来specify需要加权的特征，使用--value来specify需要加权的特征中的具体类别，使用--multiplier来specify加权的倍率，
```bash
python Algorithm/DataProcessing/weight_hard_examples.py --input_csv data_set.csv --output_csv weighted_data_set.csv --column shadow --value 1 --multiplier 4
```
训练集和测试集划分
```bash
python Algorithm/DataProcessing/split_dataset.py --input_csv data_set.csv --train_csv train_set.csv --test_csv test_set.csv --num_train 1000 --num_test 200
```
打印csv文件中所有特征的所有类别的数量
```bash
python Algorithm/DataProcessing/print_category_counts.py --file_path data_set.csv
```

### 模型
该项目使用了3种卷积神经网络模型来实现海岸线数据的训练和测试，UAED, DEXINED, MUGE，并针对该项目对UAED模型做了进一步优化，想要了解更多细节，请阅读[Algorithm Report.pdf]([https://www.unsw.edu.au/research/wrl](https://github.com/unsw-cse-comp99-3900-24t1/capstone-project-9900f16aleetcodekillers/blob/Algorithm/Algorithm/Algorithm%20Report.pdf))





