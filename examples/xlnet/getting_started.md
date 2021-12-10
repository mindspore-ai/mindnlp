# XLNet样例

## 数据集

CoLA(The Corpus of Linguistic Acceptability)

QNLI(Question NLI)

QQP(Quora Question Pairs)

SST2(The Stanford Sentiment Treebank)

## 下载地址

CoLA:https://dl.fbaipublicfiles.com/glue/data/CoLA.zip

QNLI:https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip

QQP:https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip

SST2:https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

下载并解压数据集

## 初始化权重

XLNet mindspore初始化权重由pytorch初始化权重转换而来

### mindspore初始化权重

https://pan.baidu.com/s/1S1sgwxmOewuza62kJI7hdw  提取码: 87m2

## 配置文件

上述四个数据集对应四个配置文件，每个配置文件都需要指定以下两个参数

`dataset.path #数据集文件夹路径`

`model.ckpt_path #初始化权重文件`

其他参数默认

## 开始微调与评估

```bash run.sh```

## 评估结果

权重文件在`./ckpt`目录下

评估指标在`./result/${dataset}/result.txt`文件中

性能指标在`./result/${dataset}/train_one_step_time.txt`文件中