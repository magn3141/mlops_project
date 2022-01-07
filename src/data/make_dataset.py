# -*- coding: utf-8 -*-
import click
import logging
import os
import torch
from transformers import AutoTokenizer
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('tensor_size')
@click.argument('tokenizer')
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, tensor_size, tokenizer, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed). 
        The tokenizer is used to tokenize the input data, where each data
        point is a maxium size of tensor_size. 
    """
    tensor_size = int(tensor_size)
    
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    press_conference_texts = []
    files = os.listdir(input_filepath)
    for f in files: 
        with open(input_filepath+f) as file:
            # data = file.read().replace('\n',' ')
            data = file.read()
            tokenized_text = tokenizer.encode(data, return_tensors='pt')
            start = 0 
            for i in data:
                data_point = tokenized_text[0][start:start+tensor_size]
                if (len(data_point) != tensor_size): 
                    break
                start += tensor_size 
                press_conference_texts.append(data_point)
    data_tensor = torch.stack(press_conference_texts)
    torch.save(data_tensor, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
