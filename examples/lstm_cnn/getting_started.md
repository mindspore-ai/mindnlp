# Lstm_cnn样例

## 数据集

CoNLL2003

## 下载地址

https://github.com/davidsbatista/NER-datasets/raw/master/CONLL2003

下载并解压数据集
使用python examples/lstm_cnn/conll2003/utils/conll_convert.py -s '数据集源文件夹' -t '数据集目标文件夹'
预处理数据集

## 配置文件

`dataset.path #数据集文件夹路径`

其他参数默认

```text
context：mindspore context相关参数
    mode：设置静态图或者动态图模式，0为静态图，1为动态图。
    device_target：设置目标设备，可选择CPU或者GPU。
    dataset：数据集相关参数。
    type：数据集类型。
    paths：数据集文件夹路径。
    batch_size：mini-batch大小。
    columns_list：选择输入模型的特征列。
    test_columns_list：选择输入模型的特征列。
    ps: train与test的特征列使用不一样

model：模型相关参数。
    save_path：训练后模型权重保存路径。
    result_path：自验精度结果保存路径。
```

## 开始训练与评估

在mindtext目录下

```bash example/lstm_cnn/run.sh```

## 结果

模型保存在`./lstm_cnn_model/model.ckpt`

评估指标在`./result/result.txt`