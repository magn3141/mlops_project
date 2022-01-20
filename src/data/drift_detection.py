import click
import matplotlib.pyplot as plt
import torchdrift
import torch

from sklearn.manifold import Isomap
from src.data.conference_dataset import TextDataset
from transformers import AutoConfig, AutoModelForCausalLM

@click.command()
@click.argument('working_dir', type=click.Path(exists=True))
def main(working_dir):
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    config = AutoConfig.from_pretrained(
            working_dir + "models/covid_press/config.json")
    model = AutoModelForCausalLM.from_config(config)
    regular_dataset = TextDataset(
        path=working_dir + "data/processed/data_tensor_512.pt")
    regular_dataloader = torch.utils.data.DataLoader(
        regular_dataset, batch_size=1, shuffle=True)

    torchdrift.utils.fit(regular_dataloader, model, drift_detector)
    regular_features = model(next(iter(regular_dataloader)))
    regular_score = drift_detector(regular_features)
    p_val = drift_detector.compute_p_value(regular_features)

    mapper = Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(regular_features)
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f'score {regular_score:.2f} p-value {p_val:.2f}')
    plt.savefig(working_dir + "reports/figures/regular_data_mapped.png")

    drifted_dataset = TextDataset(
        path=working_dir + "data/drifted_processed/drifted_data_tensor_512.pt")
    drifted_dataloader = torch.utils.data.DataLoader(
        drifted_dataset, batch_size=batch_size, shuffle=True)
    
    torchdrift.utils.fit(drifted_dataloader, model, drift_detector)
    drifted_features = model(next(iter(drifted_dataloader)))
    drifted_score = drift_detector(drifted_features)
    drifted_p_val = drift_detector.compute_p_value(drifted_features)

    drifted_features_embedded = mapper.transform(drifted_features)
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    plt.scatter(drifted_features_embedded[:, 0], drifted_features_embedded[:, 1], s=4)
    plt.title(f'score {drifted_score:.2f} p-value {drifted_p_val:.2f}')
    plt.savefig(working_dir + "reports/figures/drifted_data_mapped.png")


if __name__ == "__main__":
    main()