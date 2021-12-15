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

XLNet mindspore初始化权重由pytorch初始化权重转换而来。

### mindspore初始化权重

https://pan.baidu.com/s/1S1sgwxmOewuza62kJI7hdw  提取码: 87m2

## 配置文件

上述四个数据集对应四个配置文件，每个配置文件都需要指定

`dataset.path #数据集文件夹路径`

`model.ckpt_path #初始化权重文件`

其他参数默认。

`config/*_config.yaml`参数说明:

```text
context：mindspore context相关参数
    mode：设置静态图或者动态图模式，0为静态图，1为动态图。
    device_target：设置目标设备，可选择CPU或者GPU。

dataset：数据集相关参数。
    type：数据集类型。
    paths：数据集文件夹路径。
    batch_size：mini-batch大小。
    tokenizer：默认使用预训练分词器预处理数据集。
    max_length：输入模型的最大序列长度。
    truncation_strategy：截取策略，默认为true，当样本序列长度大于max_length时从序列后面开始截断到max_length长度。
    columns_list：选择输入模型的特征列（训练集）。
    test_columns_list：选择输入模型的特征列（测试集）。
    
model：模型相关参数。
    config_path：预训练模型初始化配置文件。
    ckpt_path：预训练模型初始化权重。
    save_path：训练后模型权重保存路径。
    result_path：训练、评估后训练速度、自验精度结果保存路径。
    
epoch：训练轮次。

num_labels：数据集标签种类数量。
```

## 开始微调与评估

进入examples/xlnet目录路径下。

```shell
# 训练
bash run_train.sh
```

```shell
# 评估
bash run_eval.sh
```

```shell
# 训练与评估
bash run.sh
```

## 评估结果

权重文件在`./ckpt`目录下

评估指标在`./result/${dataset}/result.txt`文件中

性能指标在`./result/${dataset}/train_one_step_time.txt`文件中