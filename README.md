# DIST-Deep-Interaction-Sentiment-Transformer

本代码为使用中关村数据集完成情感分析的具体实现，为了完整运行此代码，需要执行以下操作

## 1.下载数据集
在这一步可以选择MIMN提供的数据集，也可以选择经过处理后的数据集，如果选择MIMN提供的数据集，需要执行第2步操作，否则直接执行第3步操作。

MIMN: https://github.com/xunan0812/MIMN

或者选择经过处理后的文件(datasets文件夹)：[onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxnRe49FfF3aBpfb?e=jhp7BC)||123pan

## 2.补全缺失的信息
从中关村数据集找到multi_jsonData文件夹，将其放到项目的根目录，依此执行construct.py shuffle.py,重新构建数据集用于构建缺失的语义信息。

## 3.下载预训练模型
本研究所使用的预训练模型来自hugging face与github:

DeBERTa-ZH: https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese

RoBERTa-ZH: https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment

Swin-Transformer: https://github.com/microsoft/Swin-Transformer

根据自己的需要选择Deberta或者RoBERTa模型(pretrains文件夹)，Deberta模型相较于RoBERTa精度更高(差距在1~2%)，但显存占用与训练时长均翻倍。

## 4.下载持久化保存数据(非必要)
由于数据的预处理需要耗费大量的时间，这一步直接提供经过预处理后的数据，将其放入datasets文件夹解压即可，如果不下载持久化保存数据，请确保电脑有16G显存并等待大约1个小时完成数据的预处理。

## 5.训练
确保项目整体框架如下所示，执行main.py即可
