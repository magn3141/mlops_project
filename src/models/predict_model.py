import logging

import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


# Model_max_length
# Batch size
# Learning rate
# n_epochs


@hydra.main(config_path="config", config_name="predict")
def main(cfg: DictConfig):

    working_dir = hydra.utils.get_original_cwd()

    log.info("Loading model....")
    # Loading pretrained Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained("flax-community/dansk-gpt-wiki")
    model = AutoModelForCausalLM.from_pretrained(
        working_dir + cfg.model_relative_path)
    log.info("Model loaded.")

    # Tokenizing input and generating text
    tokenized_string = tokenizer.encode(cfg.text, return_tensors='pt')
    generated_text_encoded = model.generate(
        tokenized_string, max_length=cfg.max_length)
    generated_text = tokenizer.decode(generated_text_encoded[0])
    with open('output.txt', 'w') as f:
        f.write(generated_text)
        log.info(generated_text)


if __name__ == "__main__":
    main()
