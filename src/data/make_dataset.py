import logging
import os
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from transformers import AutoTokenizer
import csv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('tensor_size')
@click.argument('tokenizer')
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, tensor_size: str, tokenizer: str, output_filepath: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        The tokenizer is used to tokenize the input data, where each data
        point is a maximum size of tensor_size.
    """
    tensor_size = int(tensor_size)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    SUM_data = getSUMPressReleases(tokenizer, tensor_size)

    press_conference_texts = []
    files = os.listdir(input_filepath)
    for f in files:
        if f[0] == '.':
            continue
        with open(input_filepath+f) as file:
            # data = file.read().replace('\n',' ')
            # Loading data from file
            data = file.read()
            # Tokenizing data
            tokenized_text = tokenizer.encode(data, return_tensors='pt')
            start = 0
            for i in data:
                data_point = tokenized_text[0][start:start+tensor_size]
                if (len(data_point) != tensor_size):
                    break
                start += tensor_size
                press_conference_texts.append(data_point)
    # Saving data as tensor
    stm_data_tensor = torch.stack(press_conference_texts)
    SUM_data_tensor = torch.squeeze(torch.stack(SUM_data[:-1]))
    print(stm_data_tensor.shape)
    print(SUM_data_tensor.shape)

    data_tensor = torch.cat((stm_data_tensor, SUM_data_tensor), dim=0)

    print(data_tensor.shape)
    torch.save(data_tensor, output_filepath)


def getSUMPressReleases(tokenizer, split_size):
    data = None
    with open('data/raw/sum_press/data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            tokenized_text = tokenizer.encode(row[4], return_tensors='pt')
            if line_count == 0:
                data = tokenized_text
            else:
                data = torch.cat((data, tokenized_text), dim=1)
            line_count += 1
    splitted_data = torch.split(data, split_size, dim=1)
    return list(splitted_data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
