# LUKE样例

## 数据集

SQuAD1.1

## 需要准备的文件

- SQuAD1.1数据集(放置examples/luke/dataset)：1.train-v1.1.json   2.dev-v1.1.json

- enwiki文件(放置examples/luke/dataset/wiki_entity)：

https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view?usp=sharing

- vocab_entity.jsonl(放置examples/luke/dataset)

- luke_large.ckpt（初始化权重文件放至examples/luke/ckpt）

## 开始训练与评估

bash run_luke.sh

## 评估结果

权重文件在`./ckpt/luke-qa.ckpt`文件中

评估指标在`./result/predictions_.json`文件中
