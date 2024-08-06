# AI-Driven Shoreline Mapping for Coastal Monitoring
This project uses deep learning and image processing methods to automatically map coastlines, including algorithm engineering and an easy-to-use user interface. 

## Table of Contents
  
1. [User Interface](#user-interface)
    - [Key Features](#key-features)
    - [Technology Stack](#technology-stack)
    - [Installation Guide](#installation-guide)
        - [Step 1: Install Models](#step-1-install-models)
        - [Step 2: Build and Start Docker Containers](#step-2-build-and-start-docker-containers)
        - [Step 3: Access the Application](#step-3-access-the-application)
    - [Images for test](#images-for-test)
    - [Cypress](#cypress)
    - [Key File Descriptions](#key-file-descriptions)
    - [Usage Workflow](#usage-workflow)
    - [Important Notes](#important-notes)
2. [Algorithm](#algorithm)
    - [Environment Library Installation](#environment-library-installation)
    - [Data Introduction](#data-introduction)
    - [Data Preprocessing](#data-preprocessing)
    - [Model](#model)
        - [UAED](#uaed)
        - [MUGE](#muge)
        - [DEXINED](#dexined)
    - [Classification](#classification)
3. [Document](#document)

# User Interface

<img src="https://github.com/user-attachments/assets/994e6e5a-dd63-4649-a424-a554d6648a75" width="500">


## Key Features

- **Image Upload**: Support for single or multiple image uploads
- **Shoreline Detection**: Automatic detection using an advanced AI model
- **Quality Control**: Automatic image quality assessment for sample filtering
- **Result Display**: Presentation of detection results in binary and color formats
- **Result Download**: Batch download support in various formats (binary images, color images, pixel data CSV)
- **Image Manipulation**: Support for image zooming and panning
- **Logging System**: Detailed recording of all operations for easy tracking and analysis

## Technology Stack

- **Frontend**: React, Material-UI
- **Backend**: Flask
- **AI Model**: UNet++ architecture based on EfficientNet-B7
- **Image Processing**: OpenCV, NumPy
- **Others**: Pandas, Matplotlib

## Installation Guide
1. Clone the repository
```bash
git clone https://github.com/unsw-cse-comp99-3900-24t1/capstone-project-9900f16aleetcodekillers.git
cd capstone-project-9900f16aleetcodekillers
```

### Step 1: Install Models

1. **Download Models**

   Download the following models and save them in the  `backend` folder.   
   [General.pth](https://github.com/a2535171138/Shoreline-Detection/releases/download/a/General.pth)  
   [Narrabeen.pth](https://github.com/a2535171138/Shoreline-Detection/releases/download/a/Narrabeen.pth)  
   [CoastSnap.pth](https://github.com/a2535171138/Shoreline-Detection/releases/download/a/CoastSnap.pth)  
   [GoldCoast.pth](https://github.com/a2535171138/Shoreline-Detection/releases/download/a/GoldCoast.pth)  
   [coast_classifier.pth](https://github.com/a2535171138/Shoreline-Detection/releases/download/a/coast_classifier.pth)  

2. **Save Models to `backend` Folder**

   Ensure all models are downloaded and saved in the `backend` folder.

### Step 2: Build and Start Docker Containers

1. **Build Docker Images**

   Open a terminal and navigate to your project directory. Run the following command to build the Docker images:
   ```bash
   docker-compose build
   ```

2. **Start Docker Containers**

   Once the build is complete, start the Docker containers by running:
   ```bash
   docker-compose up
   ```

   This will start your application and associated services.
   Make sure you already install model

### Step 3: Access the Application

1. **Access the Local Server**

   After the Docker containers are up and running, open your browser and go to:  

   http://localhost:3000/
   

   You should now be able to use the application.

## Images for test
The images in folder `images` are provided for testing.  
<img src="https://github.com/user-attachments/assets/fc2c4308-0893-41f3-a8b2-bd659e6edc24" width="500">" 

## Cypress

   1.Before you start, make sure you have Node.js and npm installed.   
   
   2.Run the following command to install Cypress:
   ```bash
   npm install cypress --save-dev --legacy-peer-deps
   ```
   3.Use Cypress's built-in test interface to run and debug tests:
   ```bash
   npx cypress open
   ```

## Key File Descriptions

- `backend/app.py`: Main Flask backend program
- `backend/uaed_predict.py`: AI model prediction logic
- `backend/quality.py`: Image quality assessment module
- `frontend/src/App.js`: Main React frontend component
- `frontend/src/components/MiniDrawer.js`: Main interface drawer component
- `frontend/src/components/MainWorkArea.js`: Main work area component

## Usage Workflow

1. Upload Images: Click the "Upload Image" button to select and upload coastline images.
2. Quality Check: Enable the "Quality Check" feature for automatic image quality assessment.
3. Execute Detection: Click the "Get Result" button to start shoreline detection.
4. View Results: Examine detection results on the interface, toggle between binary and color views.
5. Download Results: Use the "Download All" feature to get results in desired formats.
6. View Logs: Click the "View Logs" button to see detailed operation records.

## Important Notes

- It takes a long time to get results the first time because you need to install efficientnet automatically.
- Ensure uploaded images are clear and contain distinct shorelines.
- Be patient when processing large numbers of images, as it may take some time.
- Regularly check and clear logs to maintain application performance.

# Algorithm

## Environment Library Installation
- **Python**: 3.8+
- **NumPy**: 1.23.5
- **Pandas**: 1.5.3
- **Matplotlib**: 3.7.1
- **Scikit-Image**: 0.21.0
- **SciPy**: 1.10.1
- **Pillow**: 9.5.0
- **PyTorch**: 2.0.1
- **TorchVision**: 0.15.1
- **EfficientNet-PyTorch**: 0.7.1
- **OpenCV**: 4.8.0
- **Shapely**: 2.0.1
  
## Data Introduction
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

- **Description**: The project dataset is non-public data, from [Water Research Laboratory (UNSW Sydney)](https://www.unsw.edu.au/research/wrl).  
- **Image data**: RGB coast images of varying sizes  
- **csv file**: Holds all the data used for training or testing, the *path* column is the path of each image, the *label* column is the set of coastline pixels that have been labeled, and the other columns have other feature categories of the current image that will not affect the training.  
<img src="sample.png" alt="Dataset Samples" width="500"/>  
Figure 1 - The pixel point set of label drawn on the original coast image

## Data preprocessing
In order to obtain the rectified data set with coordinate transformation `plan.csv`, please refer to the `Algorithm/jupyter notebooks/Coordinate_Correction.ipynb` file.  

Combine csv data from different scenarios and outputs them into a trainable csv file.  
```bash
python Algorithm/DataProcessing/process_dataframes.py --csv_files coastsnap_segment_clean.csv argus_goldcoast_segment.csv segment_narraV2.csv plan.csv --folders 'CoastSnap' 'Argus goldcoast' 'Argus narrabeen' --output_csv data_set.csv
```
Class balance: Use --column to specify the balanced feature, where all classes will have the same amount of data.
```bash
python Algorithm/DataProcessing/balance_dataset.py --input_csv data_set.csv --output_csv balanced_data_set.csv --column site
```
Weighting chanllenging data: Use --column to specify the feature to be weighted, --value to specify the class of the feature to be weighted, and --multiplier to specify the rate of the weighting
```bash
python Algorithm/DataProcessing/weight_hard_examples.py --input_csv data_set.csv --output_csv weighted_data_set.csv --column shadow --value 1 --multiplier 4
```
Split training and test set
```bash
python Algorithm/DataProcessing/split_dataset.py --input_csv data_set.csv --train_csv train_set.csv --test_csv test_set.csv --num_train 1000 --num_test 200
```
Print the number of all categories for all features in the csv file
```bash
python Algorithm/DataProcessing/print_category_counts.py --file_path data_set.csv
```

## Model  
This project uses 3 kinds of convolutional neural network models to realize the training and testing of coastline data. The model code from the following open source projects is used, and we would like to thank:  
DEXINED: https://github.com/xavysp/DexiNed  
UAED & MUGE: https://github.com/ZhouCX117/UAED_MuGE  

## UAED
Before using UAED for training and prediction, you need to install efficientnet-pytorch
```bash
pip install efficientnet-pytorch
```
Train
```bash
python Algorithm/UAED_MuGE/train_uaed.py --batch_size 8 --csv_path 'train_set.csv' --tmp save_path/trainval_ --warmup 5 --maxepoch 25
```
Predict: Use --value to specify the folder name to save the predictions to, and --threshold to specify the threshold for post-processing to use for binarization
```bash
python Algorithm/Test/uaed_predict.py --input_image_path 'Argus goldcoast/.../image0.jpg' --model_path 'Narrabeen.pth' --save_dir result_dir --threshold 200
```
<img src="uaed_result.png" alt="uaed_result" width="500"/>  
Figure 2 - The predicted pixel point set drawn on the original coast image  
<br>

Test: Use `--binary_threshold` to specify the threshold for post-processing to use for binarization, and `--distance_threshold` to specify the threshold for the ODS method to consider a two-point match  
```bash
python Algorithm/Test/uaed_test.py --input_csv 'test_set.csv' --model_path 'Narrabeen.pth' --save_path 'test_result.txt' --metric_method ODS --binary_threshold 200 --distance_threshold 50
```

## MUGE
Before using UAED for training and testing, you need to install openai-clip and efficientnet-pytorch
```bash
pip install openai-clip
pip install efficientnet_pytorch
```
Train
```bash
python Algorithm/UAED_MuGE/train_muge.py
```
Test
```bash
python Algorithm/Test/muge_test.py --input_csv 'test_set.csv' --model_path 'Narrabeen.pth' --save_path 'test_result.txt' --metric_method ODS --binary_threshold 200 --distance_threshold 50
```

## DEXINED
Before using DEXINED for training and testing, you need to install kornia
```bash
pip install kornia
```
Train
```bash
python DexiNed/main.py
```
Test
```bash
python Algorithm/Test/Dexined_test.py --input_csv 'test_set.csv' --model_path 'Narrabeen.pth' --save_path 'test_result.txt' --metric_method ODS --binary_threshold 200 --distance_threshold 50
```

## Classification
We have provided a pre-trained model for classifying coastlines, you can download it in [coast_classifier.pth](https://github.com/unsw-cse-comp99-3900-24t1/capstone-project-9900f16aleetcodekillers/releases/download/Models/coast_classifier.pth).  
If you want to know more details or need to train your own model, please refer to the `Algorithm/jupyter notebooks/classify.ipynb` file for reference.

# Document
We optimized the UAED model for this project. For more details, please read `Algorithm/Algorithm Report.pdf`  
To learn how to use the complete user interface, please read `User Guide.pdf` 

