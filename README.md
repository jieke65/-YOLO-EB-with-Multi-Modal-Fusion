# YOLO-EB-with-Multi-Modal-Fusion
Source code for the paper:An Enhanced Fire Perception Framework for Firefighting Robots: ECA-BiFPN Boosted YOLO-EB with Multi-Modal Fusion. Manuscript ID: IEEE LATAM Submission ID: 10447
## Author：
 - Botao Ni
 - Lei Huang
 - Ying Xiang
 - Yan Zhu
 - Lin Li
 - Yunfei Zhou
 - Jingjing Yang
 - Hao Tan
## Affiliation:
 - Botao Ni,Lei Huang and Ying Xiang are with the Guangdong University of Technology.
 - Yan Zhu and Lin Li are with the Jade Bird Fire Co., Ltd. Industry Park, Zhuoxialu Road, Zhuolu, Hebei Province.
 - Yunfei Zhou from the Guangdong Communication Polytechnic.
 - Jingjing Yang is from the School of Physics and Electronic Information, Guangxi Minzu University.
 - Hao Tang is from  the Synergy Innovation Institute of GDUT, Heyuan.

![yolov8-EB](https://github.com/user-attachments/assets/a2b1951c-1bff-4ccc-8c39-1bfd42402617)
## Project Structure
├── YOLO-EB/train.py # Training script  
├── README.md # Project documentation  
├── YOLO-EB/fire_trend # Judgment of fire trend  
├── YOLO-EB/mask_point # Segmented flame mask matching  
├── YOLO-EB/model # YAML files containing various YOLO configurations  
## Usage
Please download the dataset, and decompress it at the root folder of this repository. The dataset can be found [here](https://github.com/suyixuan123s/Fire-Segmentation-Dataset).
## Environment
a.Please run environment. yml while ensuring that the software system has Conda.
```
conda env create -f environment.yml
```
b.Please store the dataset folder and program files in the same directory.
```
.
├── data
├── model
├── my_train.py
├── environment.yml
```
## Run
```
Run my_train.py
```
