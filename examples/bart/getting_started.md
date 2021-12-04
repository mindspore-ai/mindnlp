# Bart样例

## 数据集

XSUM()

下载地址：

下载并解压

## 配置文件

`dataset.path #数据集文件夹路径`

`model.ckpt_path #初始化权重文件`

其他参数默认

## 开始训练与评估

```bash run.sh```

## 评估结果

权重文件在`./ckpt`目录下

评估指标在`./result/${dataset}/result.txt`文件中

性能指标在`./result/${dataset}/train_one_step_time.txt`文件中