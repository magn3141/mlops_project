

from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data.conference_dataset import TextDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime

# Model_max_length
# Batch size
# Learning rate
# n_epochs


@hydra.main(config_path="config", config_name="predict")
def main(cfg):

    working_dir = hydra.utils.get_original_cwd()

    tokenizer = AutoTokenizer.from_pretrained("flax-community/dansk-gpt-wiki")
    model = AutoModelForCausalLM.load_pretrained(working_dir + cfg.model_path)

    tokenized_string = tokenizer.encode(cfg.text, return_tensors='pt')
    generated_text_encoded = model.generate(
        tokenized_string, max_length=cfg.max_length)
    generated_text2 = tokenizer.decode(generated_text_encoded[0])
    print(generated_text2)


if __name__ == "__main__":
    main()
