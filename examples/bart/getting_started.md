# Bart样例

## 数据集

XSUM(Extreme Summarization)

下载地址：http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz
分割器：https://raw.githubusercontent.com/EdinburghNLP/XSum/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json

下载并解压

## 配置文件

`dataset.path #数据集文件夹路径`

`model.ckpt_path #初始化权重文件`

其他参数默认

## 开始训练与评估

```bash run.sh```

## 评估结果

权重文件在`./ckpt`目录下

评估文件在`./result/${dataset}`中

性能指标在`./result/${dataset}/train_one_step_time.txt`文件中