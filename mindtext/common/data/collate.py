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


class Pad():
    """
    Pads the input data samples to the largest length
    Args:
        pad_val (float|int, optional): The padding value. Default: 0.
        max_length (int): The required length of the data. If the input data less than
            the max_length, it indicates we pad it to the max_length with the pad_val.
        pad_right (bool, optional): The position of the PAD. If True, it indicates we
        pad to the right side, while False indicates we pad to the left side.
        Default: True.
     """

    def __init__(self, max_length, pad_val=0, pad_right=True):
        self.pad_val = pad_val
        self.max_length = max_length
        self.pad_right = pad_right

    def __call__(self, data: list):
        """
        Args:
            data (list): The input data.
        Returns:
            list: the input data which has been pad to max_length.
        Example:
            .. code-block:: python
                from mintext.common.data.collate import Pad
                pad = Pad(max_length=10)
                input_data = [1, 2, 3, 4, 5, 6]
                result = Pad(input_data)
                '''
                [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]
                '''
        """
        if self.pad_right:
            while len(data) < self.max_length:
                data.append(self.pad_val)
        else:
            while len(data) < self.max_length:
                data.insert(0, self.pad_val)
        return data
