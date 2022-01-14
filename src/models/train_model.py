import logging

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoConfig, AutoModelForCausalLM

from src.data.conference_dataset import TextDataset

# Model_max_length
# Batch size
# Learning rate
# n_epochs
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project)

    working_dir = hydra.utils.get_original_cwd()

    if (cfg.continue_training):
        # Loading a saved model
        model = AutoModelForCausalLM.from_pretrained(
            working_dir + "/models/" + cfg.name)
        config = AutoConfig.from_pretrained(
            working_dir + "/models/" + cfg.name)

        min_loss = config.to_dict()["train_run"]["min_loss"]
        log.info(
            f"Continue training on /models/{cfg.name} with loss {min_loss}")
    else:
        # Loading a pretrained backbone
        log.info("Training {cfg.backbone}")
        model = AutoModelForCausalLM.from_pretrained(cfg.backbone)
        min_loss = 10e10
    wandb.watch(model, log_freq=100)

    # Creating dataset and dataloader
    dataset = TextDataset(
        path=working_dir + "/data/processed/" + cfg.train_data)
    dataloader = torch.utils.data.DataLoader(
        dataset[:2], batch_size=cfg.batch_size, shuffle=True)
    log.info("Data loaded")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.lr)

    # Training the model
    for epoch in range(cfg.n_epochs):
        log.info(f"--- Epoch: {epoch} ---")
        running_loss = []
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    outputs = model(batch.to(device), labels=batch.to(device))

            loss = outputs[0]
            loss.backward()

            optimizer.step()

            running_loss.append(loss.item())

            if (i % 20 == 0):
                avg_loss = np.mean(running_loss)
                log.info(f"Training loss: {avg_loss}")

        epoch_loss = np.mean(running_loss)
        wandb.log({"train_loss": epoch_loss})

        # Logging and saving the model if loss is lower than previous min loss
        if (epoch_loss < min_loss):
            min_loss = epoch_loss
            # now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

            log.info(f"Saving new model with better loss ({min_loss})")
            # save both in hydra output folder & in models folder
            model.config.update(
                {"train_run": {"min_loss": 2000}})
            # min_loss
            model.save_pretrained(f"{working_dir}/models/{cfg.name}")
            model.save_pretrained("./")
        # now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
if __name__ == "__main__":
    main()
