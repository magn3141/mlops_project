import os.path

import pytest
import torch

from src.data.conference_dataset import TextDataset


# Test data size
@pytest.mark.skipif(not os.path.exists('data/processed/data_tensor_512.pt'),
                    reason="Data not found")
def test_512_length():
    train_set = TextDataset('data/processed/data_tensor_512.pt')
    assert len(train_set) == 833, "Traning dataset did not have the correct number of samples"


# Test data size
@pytest.mark.skipif(not os.path.exists('data/processed/data_tensor_1024.pt'),
                    reason="Data not found")
def test_1024_length():
    test_set = TextDataset('data/processed/data_tensor_1024.pt')
    assert len(test_set) == 411, "Testing datataset did not have the correct number of samples"


# Test data shapes
@pytest.mark.skipif(not os.path.exists('data/processed/data_tensor_512.pt'),
                    reason="Data not found")
def test_512_shapes():
    rand_tens = torch.rand(512)
    train_set = TextDataset('data/processed/data_tensor_512.pt')
    for i in train_set:
        assert i.shape == rand_tens.shape, "All training data does not have the right shape"


# Test data shapes
@pytest.mark.skipif(not os.path.exists('data/processed/data_tensor_1024.pt'),
                    reason="Data not found")
def test_1024_shapes():
    rand_tens = torch.rand(1024)
    train_set = TextDataset('data/processed/data_tensor_1024.pt')
    for i in train_set:
        assert i.shape == rand_tens.shape, "All training data does not have the right shape"
