# LUKE样例

## 论文地址

EMNLP: https://www.aclweb.org/anthology/2020.emnlp-main.523.pdf

arxiv: https://arxiv.org/pdf/2010.01057.pdf

## 数据集SQuAD1.1

数据集官网： https://rajpurkar.github.io/SQuAD-explorer/

可以通过wget方式从官网下载SQuAD1.1

'''
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
'''

## 需要准备的文件

- SQuAD1.1数据集(放置examples/luke/dataset)：1.train-v1.1.json   2.dev-v1.1.json

- enwiki文件(放置examples/luke/dataset/wiki_entity)：

https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view?usp=sharing

- vocab_entity.jsonl(放置examples/luke/dataset)

- luke_large.ckpt（初始化权重文件放至examples/luke/ckpt）

## 开始训练与评估

方法1：通过脚本来一键训练预测评估
'''
bash run_luke.sh
'''
方法2：指定yaml文件来训练以及预测评估

'''
python train.py -c config/luke_squad.yaml

python eval.py -c config/luke_squad.yaml
'''

## 评估结果

权重文件在`./ckpt/luke-qa.ckpt`文件中

评估指标在`./result/predictions_.json`文件中
