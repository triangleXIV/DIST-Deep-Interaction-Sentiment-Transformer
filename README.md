# DIST-Deep-Interaction-Sentiment-Transformer

[中文]||English
This code is a specific implementation of sentiment analysis using the Zhongguancun dataset. To fully run this code, you need to perform the following steps:

## 1.Download the Dataset
In this step, you can choose to use the dataset provided by MIMN or the processed dataset. If you choose the MIMN dataset, proceed to step 2; otherwise, go directly to step 3.

MIMN: https://github.com/xunan0812/MIMN

Or choose the processed files (datasets folder)：[onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxnRe49FfF3aBpfb?e=jhp7BC)||[123pan](https://www.123pan.com/s/f3giVv-F1l3H.html)

## 2.Complete Missing Information
Find the multi_jsonData folder from the ZOL dataset and place it in the project's root directory. Then execute construct.py and shuffle.py to rebuild the dataset for constructing missing semantic information.

## 3.Download Pre-trained Models
The pre-trained models used in this study are from Hugging Face and GitHub:

DeBERTa-ZH: https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese

RoBERTa-ZH: https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment

Swin-Transformer: https://github.com/microsoft/Swin-Transformer

Depending on your needs, choose either DeBERTa([onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxZ9oL-ZCEXMgxMA?e=YEA7Bo))||[123pan](https://www.123pan.com/s/f3giVv-B1l3H.html))or RoBERTa([onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxdO44_IGvg4Eg2F?e=O9K6ZY))||[123pan](https://www.123pan.com/s/f3giVv-J1l3H.html))model((pretrains folder)，The DeBERTa model has higher accuracy compared to RoBERTa (by 1-2%), but it requires double the GPU memory and training time.

## 4.Download Persistently Saved Data (Optional)
Since data preprocessing takes a lot of time, this step directly provides the preprocessed data. Simply extract it into the datasets folder. If you do not download the persistently saved data, ensure your computer has 16GB of GPU memory and wait about an hour to complete the data preprocessing.

DeBERTa: [onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxQQbvZzdAfy7GDP?e=eUWK3v)||[123pan](https://www.123pan.com/s/f3giVv-I1l3H.html)

RoBERTa: [onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxMqYytx4Z9BKdGm?e=n4Zeeu)||[123pan](https://www.123pan.com/s/f3giVv-w1l3H.html)

## 5.Training
Ensure the project structure is as shown below, then execute main.py:
```
├─datasets
│  ├─dev
│  │  └─swin
│  ├─img
│  ├─test
│  │  └─swin
│  └─train
│      └─swin
├─pretrains
└─swin
```

## (Optional) Training with Twitter Dataset
If you want to train with the Twitter15/17 dataset, download the Twitter dataset and rename the image folder to img, then place train, dev, and test.txt files into the datasets folder. Overwrite the original root directory files with the two .py files from the replace folder in this project.
Twitter15/17:[onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxghAlL0TnfUDZWd?e=cFmm5O)||[123an](https://www.123pan.com/s/f3giVv-w1l3H.html)
