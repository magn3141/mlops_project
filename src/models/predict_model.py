

from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data.conference_dataset import TextDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import logging
log = logging.getLogger(__name__)


# Model_max_length
# Batch size
# Learning rate
# n_epochs


@hydra.main(config_path="config", config_name="predict")
def main(cfg):

    working_dir = hydra.utils.get_original_cwd()

    log.info("Loading model....")
    tokenizer = AutoTokenizer.from_pretrained("flax-community/dansk-gpt-wiki")
    model = AutoModelForCausalLM.from_pretrained(
        working_dir + cfg.model_relative_path)
    log.info("Model loaded.")

    tokenized_string = tokenizer.encode(cfg.text, return_tensors='pt')
    generated_text_encoded = model.generate(
        tokenized_string, max_length=cfg.max_length)
    generated_text = tokenizer.decode(generated_text_encoded[0])
    with open('output.txt', 'w') as f:
        f.write(generated_text)
        log.info(generated_text)


if __name__ == "__main__":
    main()
