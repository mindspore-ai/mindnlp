import os
from mindspore.mindrecord import FileWriter
import numpy as np


def write_to_mindrecord(data, path, shared_num=1):
    """generate mindrecord"""
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    writer = FileWriter(path, shared_num)
    data_schema = {
        "src_tokens": {"type": "int32", "shape": [-1]},
        "src_tokens_length": {"type": "int32", "shape": [-1]},
        "label_idx": {"type": "int32", "shape": [-1]}
    }
    writer.add_schema(data_schema, "fasttext")
    for item in data:
        item['src_tokens'] = np.array(item['src_tokens'], dtype=np.int32)
        item['src_tokens_length'] = np.array(item['src_tokens_length'], dtype=np.int32)
        item['label_idx'] = np.array(item['label_idx'], dtype=np.int32)
        writer.write_raw_data([item])
    writer.commit()