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
├── Algorithm/  
│   └── ...  
└── ...  
csv：保存用于训练或测试的所有数据，path列是每张图像的路径，label列是已标注的海岸线像素点集，其他列为当前图像的其他特征类别，不会影响训练。  
图像数据：大小不等的RGB海岸图像  
说明：该项目数据集非公开数据，来自[Water Research Laboratory (UNSW Sydney)](https://www.unsw.edu.au/research/wrl)
<img src="sample.png" alt="Dataset Samples" width="500"/>  
Figure 1 - 将label的像素点集画在原海岸图像

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
该项目使用了3种卷积神经网络模型来实现海岸线数据的训练和测试，使用了以下开源项目中的模型代码，在此表示感谢：  
DEXINED: https://github.com/xavysp/DexiNed  
UAED & MUGE: https://github.com/ZhouCX117/UAED_MuGE  

此外，我们针对该项目对UAED模型做了进一步优化，想要了解更多细节，请阅读Algorithm Report.pdf  
Algorithm/  
└── Algorithm Report.pdf  

### UAED
在使用UAED进行训练和预测前，你需要安装efficientnet-pytorch
```bash
pip install efficientnet-pytorch
```
训练
```bash
python Algorithm/UAED_MuGE/train_uaed.py --batch_size 8 --csv_path 'train_set.csv' --tmp save_path/trainval_ --warmup 5 --maxepoch 25
```
预测，使用--value来specify保存的文件夹名，预测的结果会保存到这个文件夹中，使用--threshold来specify后处理中二值化的阈值
```bash
python Algorithm/Test/uaed_predict.py --input_image_path 'Argus goldcoast/.../image0.jpg' --model_path 'Narrabeen.pth' --save_dir result_dir --threshold 200
```
<img src="uaed_result.png" alt="uaed_result" width="500"/>  
Figure 2 - 将预测的像素点集画在原海岸图像  
测试，使用--binary_threshold来specify后处理中二值化的阈值，使用--distance_threshold来specify ODS方法中被视作两点匹配的距离阈值
```bash
python Algorithm/Test/uaed_test.py --input_csv 'test_set.csv' --model_path 'Narrabeen.pth' --save_path 'test_result.txt' --metric_method ODS --binary_threshold 200 --distance_threshold 50
```

### MUGE
在使用UAED进行训练和测试前，你需要安装openai-clip和efficientnet-pytorch
```bash
pip install openai-clip
pip install efficientnet_pytorch
```
训练
```bash
python Algorithm/UAED_MuGE/train_muge.py
```
测试
```bash
python Algorithm/Test/muge_test.py --input_csv 'test_set.csv' --model_path 'Narrabeen.pth' --save_path 'test_result.txt' --metric_method ODS --binary_threshold 200 --distance_threshold 50
```

### DEXINED
在使用DEXINED进行训练和测试前，你需要安装kornia
```bash
pip install kornia
```
训练
```bash
python DexiNed/main.py
```
测试
```bash
python Algorithm/Test/Dexined_test.py --input_csv 'test_set.csv' --model_path 'Narrabeen.pth' --save_path 'test_result.txt' --metric_method ODS --binary_threshold 200 --distance_threshold 50
```
