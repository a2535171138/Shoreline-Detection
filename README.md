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
├── data_set.csv  
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
打算
