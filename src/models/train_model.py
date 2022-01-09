from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data.conference_dataset import TextDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import numpy as np 

# Model_max_length
# Batch size
# Learning rate
# n_epochs


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):

    working_dir = hydra.utils.get_original_cwd()

    model = AutoModelForCausalLM.from_pretrained(cfg.backbone)

    dataset = TextDataset(path=working_dir + "/data/processed/" + cfg.train_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.lr)
    for epoch in range(cfg.n_epochs):
        print("--- Epoch:", epoch, "---")
        running_loss = []
        for i, batch in enumerate(dataloader):
            outputs = model(batch, labels=batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.item())
            if (i%20 == 0):        
                avg_loss = np.mean(running_loss)
                print(f"Training loss: {avg_loss}")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model.save_pretrained(working_dir + "/models", cfg.name + "_"+now)


if __name__ == "__main__":
    main()
