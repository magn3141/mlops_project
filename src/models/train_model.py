from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data.conference_dataset import TextDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import numpy as np 
import wandb

# Model_max_length
# Batch size
# Learning rate
# n_epochs


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    wandb.init(entity="mlops_grp40", project="danish_gpt2")

    working_dir = hydra.utils.get_original_cwd()

    model = AutoModelForCausalLM.from_pretrained(cfg.backbone)
    wandb.watch(model, log_freq=100)

    dataset = TextDataset(path=working_dir + "/data/processed/" + cfg.train_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.lr)
    min_loss = 1

    for epoch in range(cfg.n_epochs):
        print("--- Epoch:", epoch, "---")
        running_loss = []
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()

            outputs = model(batch.to(device), labels=batch.to(device))

            loss = outputs[0]
            loss.backward()

            optimizer.step()

            running_loss.append(loss.item())

            if (i%20 == 0):        
                avg_loss = np.mean(running_loss)
                print(f"Training loss: {avg_loss}")

        epoch_loss = np.mean(running_loss)
        wandb.log({"train_loss": epoch_loss})

        if (epoch_loss < min_loss):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model.save_pretrained(working_dir + "/models", cfg.name + "_"+now)


if __name__ == "__main__":
    main()
