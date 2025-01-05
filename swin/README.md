# DIST-Deep-Interaction-Sentiment-Transformer


中文||[English](https://github.com/triangleXIV/DIST-Deep-Interaction-Sentiment-Transformer/blob/main/README.md)

本github是论文《Breaking Through Misconceptions: The Leap of Transformer on the Chinese ACMSC Dataset》的具体实现，不过为了突出模型的多语言，多领域性，之后会补充一个越南语的私有ACMSC数据集。

事实上，写这篇论文的时候我顶着巨大的压力，完成了期刊的投稿(要是再被拒稿, 我有极大可能会面临延毕的风险Orz)，不过好在主编和审稿人都非常的友善，给我这个论文提了不少建议，也给了我很大的帮助，不胜感激。
<div align="center">
<img src="https://github.com/triangleXIV/DIST-Deep-Interaction-Sentiment-Transformer/blob/main/swin/1.jpg" alt="Image text" width="25%">
</div>
简单来说，我认为我的这篇论文算是对[HIMT](https://github.com/NUSTM/HIMT)模型的增量研究，我在阅读这篇论文的时候，发现他使用了Multi-ZOL和Twitter一共三个数据集，明明HIMT这个模型在Multi-ZOL这个中文数据集上提升比英文的Twitter数据集提升要显著，然而这篇论文和相关的代码却着重分析的是Twitter这个英文数据集，所以有了一丝灵感的我，决定顺着这个论文的思路，完成中文的Multi-ZOL数据集相关研究，于是有了这篇论文。

首先，我找到了Multi-ZOL数据集（没有标点）然后开始训练，不管是八个周期还是十个周期，按照我的模型跑，指标几乎是不变的，64-66%左右，（论文里写低了，一开始跑得时候可能运气太差了，后面模型又跑了几次，没有标点就是64-66%左右）。

我这时发现Multi-ZOL的txt文件缺乏了标点符号，要知道在分词器里，标点符号是会进行编码的，于是我找到了原始的json数据，加上了标点。然后按照HIMT模型的思路，训练了8个周期，效果是和HIMT很接近的，其实也还是66%左右，但是我多跑了几次，指标突然产生了一次飞跃，直接跑到了70多，这时我把训练周期从8改到了10，结果可以稳定在70多，也就是说，加了标点符号以后，模型在第八个周期以后还能继续提升性能，这就很神奇了。

然后就是很正常的消融实验，把HIMT模型改了改，然后这个HIMT的ARM模块（计算交叉注意力前后损失）我感觉是没啥用的，也许是只对原始的BERT有用，或者只在缺乏语义信息的时候，利用这个额外损失把结果给蒙对，也许在某些时候有奇效，但是对我的模型是没用的，所以我把ARM删掉了。

最后，这个模型其实主要是用于一句话对应多张图片的情况的，所以对于Twitter数据集这种一对一的效果会差一些，目前对于Twitter数据集，最优解应该是生成式模型，即传入文本和图像，然后让模型生成一个回答："[某个实体] 是 [MASK] 的"。其中MASK是模型将要生成的情感词汇。如果单纯想在Twitter数据集进行研究，建议不要采用我的模型。

为了完整运行此代码，需要执行以下操作：

## 1.下载数据集
在这一步可以选择MIMN提供的数据集，也可以选择经过处理后的数据集，如果选择MIMN提供的数据集，需要执行第2步操作，否则直接执行第3步操作。

MIMN: https://github.com/xunan0812/MIMN

或者选择经过处理后的文件(datasets文件夹)：[onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxnRe49FfF3aBpfb?e=jhp7BC)||[123pan](https://www.123pan.com/s/f3giVv-F1l3H.html)

## 2.补全缺失的信息
从中关村数据集找到multi_jsonData文件夹，将其放到项目的根目录，依此执行construct.py shuffle.py,重新构建数据集用于构建缺失的语义信息。

## 3.下载预训练模型
本研究所使用的预训练模型来自hugging face与github:

DeBERTa-ZH: https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese

RoBERTa-ZH: https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment

Swin-Transformer: https://github.com/microsoft/Swin-Transformer

根据自己的需要选择DeBERTa([onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxZ9oL-ZCEXMgxMA?e=YEA7Bo))||[123pan](https://www.123pan.com/s/f3giVv-B1l3H.html))或者RoBERTa([onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxdO44_IGvg4Eg2F?e=O9K6ZY))||[123pan](https://www.123pan.com/s/f3giVv-J1l3H.html))模型(pretrains文件夹)，Deberta模型相较于RoBERTa精度更高(差距在1~2%)，但显存占用与训练时长均翻倍。

## 4.下载持久化保存数据(非必要)
由于数据的预处理需要耗费大量的时间，这一步直接提供经过预处理后的数据，将其放入datasets文件夹解压即可，如果不下载持久化保存数据，请确保电脑有16G显存并等待大约1个小时完成数据的预处理。

DeBERTa: [onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxQQbvZzdAfy7GDP?e=eUWK3v)||[123pan](https://www.123pan.com/s/f3giVv-I1l3H.html)

RoBERTa: [onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxMqYytx4Z9BKdGm?e=n4Zeeu)||[123pan](https://www.123pan.com/s/f3giVv-w1l3H.html)

## 5.训练
确保项目整体框架如下所示，执行main.py即可
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

## (额外)Twitter数据集训练
如果要进行Twitter15/17数据集的训练，下载Twitter数据集并将图片文件夹改名为img,并且将train，dev，test.txt文件放入datasets文件夹,用本项目的replace文件夹里的两个py文件覆盖原本的根目录文件即可。
Twitter15/17:[onedrive](https://1drv.ms/u/s!Akl56EV1csnmoxghAlL0TnfUDZWd?e=cFmm5O)||[123an](https://www.123pan.com/s/f3giVv-w1l3H.html)

