# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Test Trainer run function"""
import unittest
import numpy as np

import mindspore.nn as nn
import mindspore.dataset as ds

from mindtext.engine.trainer import Trainer
from mindtext.common.metrics import Accuracy
from mindtext.engine.callbacks.timer_callback import TimerCallback
from mindtext.engine.callbacks.earlystop_callback import EarlyStopCallback
from mindtext.engine.callbacks.best_model_callback import BestModelCallback
from mindtext.engine.callbacks.checkpoint_callback import CheckpointCallback

np.random.seed(1)

class MyDataset:
    """Dataset"""
    def __init__(self):
        self.data = np.random.randn(20, 3).astype(np.float32)
        self.label = list(np.random.choice([0, 1]).astype(np.float32) for i in range(20))
        self.length = list(np.random.choice([0, 1]).astype(np.float32) for i in range(20))
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.length[index]
    def __len__(self):
        return len(self.data)

class MyModel(nn.Cell):
    """Model"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 1)
    def construct(self, data):
        output = self.fc(data)
        return output

class MyModel2(nn.Cell):
    """Model2"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 1)
    def construct(self, data, label, length):
        output = self.fc(data)
        label = label + label + length
        return output

class TestTrainerRun(unittest.TestCase):
    r"""
    Test Trainer Run
    """
    def setUp(self):
        self.input = None
        # 1. define dataset
        self.dataset_generator = MyDataset()
        self.train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        self.eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        # 2. define Models & Loss & Optimizer
        self.net = MyModel()
        self.net_2 = MyModel2()
        self.net.update_parameters_name('net.')
        self.net_2.update_parameters_name('net_2.')

        self.loss_fn = nn.MSELoss()
        self.optimizer = nn.Adam(self.net.trainable_params(), learning_rate=0.01)
        # 3. define callbacks
        self.timer_callback_epochs = TimerCallback(print_steps=-1)
        self.earlystop_callback = EarlyStopCallback(patience=2)
        self.bestmodel_callback = BestModelCallback(save_path='save/callback/best_model', auto_load=True)
        self.checkpoint_callback = CheckpointCallback(save_path='save/callback/ckpt_files', epochs=1)
        self.callbacks = [self.timer_callback_epochs, self.earlystop_callback, self.bestmodel_callback]
        # 4. define metrics
        self.metric = Accuracy()
        # 5. define trainer
        self.pure_trainer = Trainer(network=self.net, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                                    metrics=self.metric, epochs=2, batch_size=4, optimizer=self.optimizer,
                                    loss_fn=self.loss_fn)

    def test_pure_trainer_pynative(self):
        # 6. trainer run
        self.pure_trainer.run(mode='pynative', tgt_columns='label')

    def test_pure_trainer_graph(self):
        self.pure_trainer.run(mode='graph', tgt_columns='label')

    def test_trainer_timer_pynative(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=2, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.timer_callback_epochs)
        trainer.run(mode='pynative', tgt_columns='label')

    def test_trainer_timer_graph(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=2, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.timer_callback_epochs)
        trainer.run(mode='graph', tgt_columns='label')

    def test_trainer_earlystop_pynative(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=6, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.earlystop_callback)
        trainer.run(mode='pynative', tgt_columns='label')

    def test_trainer_earlystop_graph(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=6, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.earlystop_callback)
        trainer.run(mode='graph', tgt_columns='label')

    def test_trainer_bestmodel_pynative(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=4, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.bestmodel_callback)
        trainer.run(mode='pynative', tgt_columns='label')

    def test_trainer_bestmodel_graph(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=4, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.bestmodel_callback)
        trainer.run(mode='graph', tgt_columns='label')

    def test_trainer_checkpoint_pynative(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=6, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.checkpoint_callback)
        trainer.run(mode='pynative', tgt_columns='label')

    def test_trainer_checkpoint_graph(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net, train_dataset=train_dataset, eval_dataset=eval_dataset, metrics=self.metric,
                          epochs=6, batch_size=4, optimizer=self.optimizer, loss_fn=self.loss_fn,
                          callbacks=self.checkpoint_callback)
        trainer.run(mode='graph', tgt_columns='label')

    def test_different_model(self):
        train_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        eval_dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label", "length"], shuffle=False)
        trainer = Trainer(network=self.net_2, train_dataset=train_dataset, eval_dataset=eval_dataset,
                          metrics=self.metric, epochs=2, batch_size=4, optimizer=self.optimizer,
                          loss_fn=self.loss_fn)
        trainer.run(mode='graph', tgt_columns='length')
