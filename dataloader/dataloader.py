import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

__all__ = ['MyDataLoader', 'PreFetcher']


class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class MyDataLoader(object):
    r""" PyTorch DataLoader for COST2100 dataset.
    """

    def __init__(self, train_path, val_path, test_path, batch_size, num_workers, pin_memory, channel=2, nt=32, nc=32):
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.exists(test_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.channel = channel
        self.nt = nt
        self.nc = nc
        
        # Training data loading
        data_train = torch.load(train_path, weights_only=False, map_location=torch.device('cpu'))
        self.train_dataset = TensorDataset(data_train)

        # Validation data loading
        data_val = torch.load(val_path, weights_only=False, map_location=torch.device('cpu'))
        self.val_dataset = TensorDataset(data_val)

        # Test data loading
        data_test = torch.load(test_path, weights_only=False, map_location=torch.device('cpu'))
        self.test_dataset = TensorDataset(data_test)
        

        # import scipy.io as sio
        # # Training data loading
        # data_train = sio.loadmat(train_path)['HT']
        # data_train = torch.tensor(data_train, dtype=torch.float32).view(
        #     data_train.shape[0], self.channel, self.nt, self.nc)
        # self.train_dataset = TensorDataset(data_train)

        # # Validation data loading
        # data_val = sio.loadmat(val_path)['HT']
        # data_val = torch.tensor(data_val, dtype=torch.float32).view(
        #     data_val.shape[0], self.channel, self.nt, self.nc)
        # self.val_dataset = TensorDataset(data_val)

        # # Test data loading
        # data_test = sio.loadmat(test_path)['HT']
        # data_test = torch.tensor(data_test, dtype=torch.float32).view(
        #     data_test.shape[0], self.channel, self.nt, self.nc)
        # self.test_dataset = TensorDataset(data_test)


    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            val_loader = PreFetcher(val_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, val_loader, test_loader
