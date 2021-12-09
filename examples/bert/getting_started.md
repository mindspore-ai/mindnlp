# Bert样例

## 数据集

CoLA(The Corpus of Linguistic Acceptability)

AFQMC(Ant Financial Question Matching Corpus)

TNEWS(Short Text Classificaiton for News)

IFLYTEK(Long Text classification)

OCNLI(Original Chinese Natural Language Inference)

CMNLI(Chinese Multi-Genre NLI)

LCQMC(A Large-scale Chinese Question Matching Corpus)

下载地址：https://gluebenchmark.com/tasks
         https://github.com/CLUEbenchmark/CLUE
         http://icrc.hitsz.edu.cn/info/1037/1146.htm


下载并解压

## 配置文件

上述七个数据集对应七个配置文件，每个配置文件都需要指定

`dataset.path #数据集文件夹路径`

`dataset.tokenizer #指定分词器`

`model.ckpt_path #初始化权重文件`

其他参数默认

## 训练示例

python bert_train.py -c ./config/cola_config.yaml，其中./config/cola_config.yaml用来指定对应的下游任务，这里指定了CoLA任务。

## 评估示例

python bert_eval.py -c ./config/cola_config.yaml，其中./config/cola_config.yaml用来指定对应的下游任务，这里指定了CoLA任务。

## 评估结果

权重文件在`./result/${dataset}/`目录下

评估指标在`./result/${dataset}/result.txt`文件中

性能指标在`./result/${dataset}/train_one_step_time.txt`文件中
