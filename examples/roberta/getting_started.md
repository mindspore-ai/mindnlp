# Roberta样例

## 数据集

CoLA(The Corpus of Linguistic Acceptability)

MNLI(Multi-Genre Natural Language Inference)

MRPC(Microsoft Research Paraphrase Corpus)

QNLI(Question Natural Language Inference)

QQP(Quora Question Pairs)

SST2(The Stanford Sentiment Treebank)

## 下载地址

CoLA:https://dl.fbaipublicfiles.com/glue/data/CoLA.zip

MNLI:https://dl.fbaipublicfiles.com/glue/data/MNLI.zip

MRPC:https://www.microsoft.com/en-us/download/confirmation.aspx?id=52398

MRPC_dev_ids([验证集ids点击此处下载](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc))

QNLI:https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip

QQP:https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip

SST2:https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

下载并解压数据集

## 初始化权重

Roberta mindspore初始化权重由pytorch初始化权重转换而来。

### mindspore初始化权重



## 配置文件

上述六个数据集对应六个配置文件，每个配置文件都需要指定

`dataset.path #数据集文件夹路径`

`model.ckpt_path #初始化权重文件`

其他参数默认

`config/*_config.yaml`参数说明:

```text
context：mindspore context相关参数
    mode：设置静态图或者动态图模式，0为静态图，1为动态图。
    device_target：设置目标设备，可选择CPU或者GPU。
    dataset：数据集相关参数。

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

## 开始训练与评估

进入examples/roberta目录路径下。

```shell
# 训练
bash run_train.sh
```

```shell
# 评估 CoLA、MRPC、QNLI、QQP、SST2
bash run_eval.sh
# 评估 MNLI
bash run_mnlieval.sh
```

```shell
# 训练
bash run.sh
```

## 评估结果

权重文件在`./ckpt`目录下
 
评估指标在`./result/${dataset}/result.txt`文件中

性能指标在`./result/${dataset}/train_one_step_time.txt`文件中