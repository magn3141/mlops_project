import os
import re
import click

def drift_data(data: str):
    """Drifts input string by replacing words
       with previous words to change the distribution
       of words.
    """
    keep_chars = ['.',',','!','?']
    split_data = re.findall(r"[\w']+|[.,!?;]", data)
    for i in range(len(split_data)):
        if i%5 == 0:
            if split_data[i] not in keep_chars and split_data[i-1] not in keep_chars: 
                split_data[i] = split_data[i-1]
    drifted_data = " ".join(split_data)
    return drifted_data


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Drifts existing dataset to test model
       for robustness towards data drifting.
    """
    files = os.listdir(input_filepath)
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)

    for f in files:
        with open(input_filepath+f, encoding="utf-8") as file:
            data = file.read()
            drifted_data = drift_data(data)
            # Creating new txt file with the drifted data
            drifted_file = open(output_filepath+"drifted_"+f, "w")
            drifted_file.write(drifted_data)
            drifted_file.close()


if __name__ == '__main__':
    main()