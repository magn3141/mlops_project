import os

import pytest
import torch
from transformers import AutoModelForCausalLM

inputs = [(torch.randint(8000, (1, 512)), torch.rand(50257)),
          (torch.randint(8000, (4, 512)), torch.rand(50257)),
          (torch.randint(8000, (8, 512)), torch.rand(50257))]


@pytest.mark.skipif(not os.path.exists('models/'), reason="No models found")
@pytest.mark.parametrize("test_input, expected", inputs)
def test_output_shape(test_input: torch.Tensor, expected: torch.Tensor):
    # Loading pretrained model
    model = AutoModelForCausalLM.from_pretrained("flax-community/dansk-gpt-wiki")
    '''
    We test that independent of the batch size the last tensor in the output
    is always the same size.
    '''
    y = model(test_input)
    assert y[0][0][-1].shape == expected.shape, "The output shape was not as expected"
