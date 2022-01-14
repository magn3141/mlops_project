import os

import numpy as np
import pytest
import torch
from torch import optim
from transformers import AutoModelForCausalLM

from src.data.conference_dataset import TextDataset


@pytest.mark.skipif(not os.path.exists('models/'), reason="No models found")
@pytest.mark.skipif(not os.path.exists('data/processed/data_tensor_512.pt'),
                    reason="Data not found")
def test_gradients():
    '''
    In the test below we test whether the gradients are updated after
    backpropogating on our training data.

    '''
    # Loading a pretrained model
    model = AutoModelForCausalLM.from_pretrained("flax-community/dansk-gpt-wiki")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-5)

    train_set = TextDataset('data/processed/data_tensor_512.pt')

    optimizer = optim.Adam(model.parameters(), lr=float(3e-5))

    text = train_set[0]

    # Backpropagation and stepping optimizer
    outputs = model(text, labels=text)
    loss = outputs[0]
    loss.backward()

    optimizer.step()

    gradients = model.transformer.h[11].mlp.c_fc.weight.grad.view(-1)

    assert len(gradients.unique()) > 1, "The gradients were not updated after backpropping"


@pytest.mark.skipif(not os.path.exists('models/'), reason="No models found")
@pytest.mark.skipif(not os.path.exists('data/processed/data_tensor_512.pt'),
                    reason="Data not found")
def test_loss_reduction():
    '''
    In the test below we test whether the loss is reduced after a very
    small training session.

    '''
    # Loading a pretrained model
    model = AutoModelForCausalLM.from_pretrained("flax-community/dansk-gpt-wiki")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-5)

    dataset = TextDataset('data/processed/data_tensor_512.pt')
    dataloader = torch.utils.data.DataLoader(dataset[:2], batch_size=1, shuffle=True)

    # Short training loop
    running_loss = []
    for epoch in range(2):
        running_loss_e = []
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(batch, labels=batch)

            loss = outputs[0]
            loss.backward()

            optimizer.step()

            running_loss_e.append(loss.item())
        running_loss.append(np.mean(running_loss_e))
    assert running_loss[0] > running_loss[-1]
