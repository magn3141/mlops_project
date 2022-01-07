from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data.conference_dataset import TextDataset
import hydra
from omegaconf import DictConfig, OmegaConf


# Model_max_length
# Batch size
# Learning rate
# n_epochs

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    working_dir = hydra.utils.get_original_cwd()

    tokenizer = AutoTokenizer.from_pretrained(
        "flax-community/dansk-gpt-wiki", model_max_length=cfg.model_max_length)
    model = AutoModelForCausalLM.from_pretrained(
        "flax-community/dansk-gpt-wiki")

    dataset = TextDataset(path=working_dir + "/data/raw/press_conferences/",
                          tokenizer=tokenizer, max_length=cfg.model_max_length)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.lr)
    for epoch in range(cfg.n_epochs):
        print("--- Epoch:", epoch, "---")
        running_loss = 0
        for i, batch in enumerate(dataloader):
            outputs = model(batch, labels=batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        avg_loss = running_loss/len(dataloader)
        print(f"Epoch training loss: {loss}")

    model.save_pretrained(working_dir + "/models")


if __name__ == "__main__":
    main()
