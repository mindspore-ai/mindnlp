"""squad for luke model"""
import json
import os
import joblib
import numpy as np
from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds
from transformers import AutoTokenizer

from .feature import convert_examples_to_features
from .dataset import SquadV1Processor
from .entity_vocab import EntityVocab
from .wiki_link_db import WikiLinkDB


class SquadLuke:
    """
    Squad process for luke model
    """
    def __call__(self, evaluate=False, config=None) -> ds.MindDataset:
        if os.path.exists(config.SQUAD_TRAIN_MR) and not evaluate:

            data_set = ds.MindDataset(dataset_file=config.SQUAD_TRAIN_MR, columns_list=config.columns_list)
            data_set = data_set.batch(config.batch_size)

        elif os.path.exists(config.SQUAD_DEV_MR) and evaluate:
            data_set = ds.MindDataset(dataset_file=config.SQUAD_DEV_MR, columns_list=config.test_columns_list)
            data_set = data_set.batch(config.eval_batch_size)
        else:
            dict_list = self.process(config, evaluate=evaluate)
            self._write_to_mr(dict_list, evaluate, config)
            if evaluate:
                data_set = ds.MindDataset(dataset_file=config.SQUAD_DEV_MR, columns_list=config.test_columns_list)
                data_set = data_set.batch(config.eval_batch_size)
            else:
                data_set = ds.MindDataset(dataset_file=config.SQUAD_TRAIN_MR, columns_list=config.columns_list)
                data_set = data_set.batch(config.batch_size)

        return data_set

    def process(self, config, evaluate):
        """

        Args:
            config:
            evaluate:

        Returns:

        """
        processor = SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(config.path)
        else:
            examples = processor.get_train_examples(config.path)
        segment_b_id = 1
        add_extra_sep_token = False

        model_redirect_mappings = joblib.load(config.wiki_entity_path + "enwiki_20181220_redirects.pkl")
        link_redirect_mappings = joblib.load(config.wiki_entity_path + "enwiki_20160305_redirects.pkl")
        tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large")
        wiki_link_db = WikiLinkDB(config.wiki_entity_path + "enwiki_20160305.pkl")

        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            entity_vocab=EntityVocab(config.entity_vocab_path),
            wiki_link_db=wiki_link_db,
            model_redirect_mappings=model_redirect_mappings,
            link_redirect_mappings=link_redirect_mappings,
            max_seq_length=512,
            max_mention_length=30,
            doc_stride=128,
            max_query_length=64,
            min_mention_link_prob=0.01,
            segment_b_id=segment_b_id,
            add_extra_sep_token=add_extra_sep_token,
            is_training=True
        )
        json_features = []
        for item in features:
            js = json.dumps(item.__dict__)
            json_features.append(js)
        arr_features = np.array(json_features)

        return arr_features

    def _write_to_mr(self, arr_features, evaluate, config):
        """
        Write RCDataset to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (str): Path of mindrecord file.
            process_function (callable): A function is used to preprocess data.

        Returns:
            List[str]: Dataset field
        """
        list_dict = []
        for item in arr_features:
            dict_temp = json.loads(item)
            list_dict.append(dict_temp)

        if not evaluate:
            squad_mindrecord_file = config.SQUAD_TRAIN_MR
            list_dict = self.padding(list_dict)

            if os.path.exists(squad_mindrecord_file):
                os.remove(squad_mindrecord_file)
                os.remove(squad_mindrecord_file + ".db")

            writer = FileWriter(file_name=squad_mindrecord_file, shard_num=1)

            data_schema = {
                "word_ids": {"type": "int32", "shape": [-1]},
                "word_segment_ids": {"type": "int32", "shape": [-1]},
                "word_attention_mask": {"type": "int32", "shape": [-1]},
                "entity_ids": {"type": "int32", "shape": [-1]},
                "entity_position_ids": {"type": "int32", "shape": [24, 24]},  #
                "entity_segment_ids": {"type": "int32", "shape": [-1]},
                "entity_attention_mask": {"type": "int32", "shape": [-1]},
                "start_positions": {"type": "int32", "shape": [-1]},
                "end_positions": {"type": "int32", "shape": [-1]}
            }
            writer.add_schema(data_schema, "it is a preprocessed squad dataset")

            data = []
            i = 0
            for item in list_dict:
                i += 1
                sample = {
                    "word_ids": np.array(item["word_ids"], dtype=np.int32),
                    "word_segment_ids": np.array(item["word_segment_ids"], dtype=np.int32),
                    "word_attention_mask": np.array(item["word_attention_mask"], dtype=np.int32),
                    "entity_ids": np.array(item["entity_ids"], dtype=np.int32),
                    "entity_position_ids": np.array(item["entity_position_ids"], dtype=np.int32),
                    "entity_segment_ids": np.array(item["entity_segment_ids"], dtype=np.int32),
                    "entity_attention_mask": np.array(item["entity_attention_mask"], dtype=np.int32),
                    "start_positions": np.array(item["start_positions"], dtype=np.int32),
                    "end_positions": np.array(item["end_positions"], dtype=np.int32),
                }

                data.append(sample)
                # print(sample)
                if i % 10 == 0:
                    writer.write_raw_data(data)
                    data = []
            print(data[0])
            if data:
                writer.write_raw_data(data)

            writer.commit()
        else:
            squad_mindrecord_file = config.SQUAD_DEV_MR
            list_dict = self.padding(list_dict)

            if os.path.exists(squad_mindrecord_file):
                os.remove(squad_mindrecord_file)
                os.remove(squad_mindrecord_file + ".db")

            writer = FileWriter(file_name=squad_mindrecord_file, shard_num=1)

            data_schema = {
                "unique_id": {"type": "int32", "shape": [-1]},
                "word_ids": {"type": "int32", "shape": [-1]},
                "word_segment_ids": {"type": "int32", "shape": [-1]},
                "word_attention_mask": {"type": "int32", "shape": [-1]},
                "entity_ids": {"type": "int32", "shape": [-1]},
                "entity_position_ids": {"type": "int32", "shape": [-1, 24]},
                "entity_segment_ids": {"type": "int32", "shape": [-1]},
                "entity_attention_mask": {"type": "int32", "shape": [-1]},
            }
            writer.add_schema(data_schema, "it is a preprocessed squad dataset")

            data = []
            i = 0
            for item in list_dict:
                i += 1
                sample = {
                    "unique_id": np.array(item["unique_id"], dtype=np.int32),
                    "word_ids": np.array(item["word_ids"], dtype=np.int32),
                    "word_segment_ids": np.array(item["word_segment_ids"], dtype=np.int32),
                    "word_attention_mask": np.array(item["word_attention_mask"], dtype=np.int32),
                    "entity_ids": np.array(item["entity_ids"], dtype=np.int32),
                    "entity_position_ids": np.array(item["entity_position_ids"], dtype=np.int32),
                    "entity_segment_ids": np.array(item["entity_segment_ids"], dtype=np.int32),
                    "entity_attention_mask": np.array(item["entity_attention_mask"], dtype=np.int32),
                }

                data.append(sample)
                if i % 10 == 0:
                    writer.write_raw_data(data)
                    data = []

            if data:
                writer.write_raw_data(data)

            writer.commit()

    def padding(self, list_dict):
        """

        Args:
            list_dict:

        Returns:

        """
        pad = lambda a, i: a[0:i] if len(a) > i else a + [0] * (i - len(a))
        pad1 = lambda a, i: a[0:i] if len(a) > i else a + [1] * (i - len(a))
        pad_entity = lambda a, i: a[0:i] if len(a) > i else np.append(a, [-1] * (i - len(a)))
        for slist in list_dict:
            slist["entity_attention_mask"] = pad(slist["entity_attention_mask"], 24)
            slist["entity_ids"] = pad(slist["entity_attention_mask"], 24)
            slist["entity_segment_ids"] = pad(slist["entity_segment_ids"], 24)

            slist["word_ids"] = pad1(slist["word_ids"], 512)
            slist["word_segment_ids"] = pad(slist["word_segment_ids"], 512)
            slist["word_attention_mask"] = pad(slist["word_attention_mask"], 512)
            # entity padding 1
            entity_size = len(slist["entity_position_ids"])
            slist["entity_position_ids"] = np.array(slist["entity_position_ids"])
            temp = [[-1] * 24 for i in range(24)]
            for i in range(24):
                if i < entity_size - 1:
                    temp[i] = (pad_entity(slist["entity_position_ids"][i], 24))

            slist["entity_position_ids"] = temp
        return list_dict
