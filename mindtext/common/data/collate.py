# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
PAD operation
"""
from typing import Union, List


def _get_bucket_length(x, bts):
    x_len = len(x)
    for index in range(1, len(bts)):
        if bts[index - 1] < x_len <= bts[index]:
            return bts[index]
    return bts[0]


class Pad:
    """
    Pads the input data samples to the largest length.

    Args:
        max_length (int): The required length of the data. If the input data less than
            the max_length, it indicates we pad it to the max_length with the pad_val.
        pad_val (Union[float, int]): The padding value, default 0.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.
        pad_right (bool): The position of the PAD. If True, it indicates we
        pad to the right side, while False indicates we pad to the left side, default True.

     """

    def __init__(self, max_length: int = 0, pad_val: Union[float, int] = 0, buckets: List[int] = None,
                 pad_right: bool = True):
        self.pad_val = pad_val
        self.max_length = max_length
        self.buckets = buckets
        self.pad_right = pad_right

    def __call__(self, data: List[int]) -> List[int]:
        """
        Args:
            data (List[int]): The input data.

        Returns:
            List[int]: The input data which has been pad to max_length.

        Examples:
            .. code-block:: python
            >>> pad = Pad(max_length=10)
            >>> input_data = [1, 2, 3, 4, 5, 6]
            >>> result = Pad(input_data)
            >>> print(result)
            (1,2,3,4,5,6,0,0,0,0)
        """
        if self.buckets is not None:
            self.max_length = _get_bucket_length(data, self.buckets)
        if self.pad_right:
            data = data + (self.max_length - len(data)) * [self.pad_val]
        else:
            data = (self.max_length - len(data)) * [self.pad_val] + data
        return data

    @staticmethod
    def padding(data: List[int], max_length: int, pad_val: Union[float, int] = 0, pad_right: bool = True) -> List[int]:
        """
        Args:
            data (List[int]): The input data.
            max_length (int): The required length of the data. If the input data less than
                the max_length, it indicates we pad it to the max_length with the pad_val.
            pad_val (Union[float, int]): The padding value, default 0.
            pad_right (bool): The position of the PAD. If True, it indicates we
            pad to the right side, while False indicates we pad to the left side, default True.

        Returns:
            List[int]: the input data which has been pad to max_length.
        """
        if pad_right:
            data = data + (max_length - len(data)) * [pad_val]
        else:
            data = (max_length - len(data)) * [pad_val] + data
        return data
