mlops_project
==============================
This project is part of the 02476 Machine Learning Operations Course (https://skaftenicki.github.io/dtu_mlops/).

## Project Description

The scope of this project is to work with the topics from the course by fine-tuning an NLP model. We want to fine-tune an NLP model to generate Covid-19 press conference transcriptions from an input sentence. The goal is to generate press conference transcriptions that seem as genuine as possible.

We will be using the Transformers framework from the huggingface community. The Transformers framework provides access to many pretrained models including many for NLP. This will allow us to use pretrained models which can be fine-tuned to our specific goal.

Initially, we will fine-tune the model using the official transcripts from the Prime Minister's Office (Statsministeriet) press conferences. Later we will look into including press releases from other offices. The press conference transcripts are acquired through official websites and initially saved as plaintext. The data will then be preprocessed to allow for training and fine-tuning of models. The dataset will consist of 31 different transcripts concerning COVID-19 and other matters of national and international importance. These are processed into Tensors and saved to allow for the use of all the data from the press conference transcripts as the size of the training data that a model can take at once is limited.

We expect to work on fine-tuning the GPT-2 model using the aforementioned dataset. GPT-2 has been fine-tuned and used for similar purposes before.

### Demo: https://magn3141.github.io/mlops_project/

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## Project checklist

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed very
point on the checklist for the exam.

### Week 1

- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [x] Create a dedicated environment for you project to keep track of your packages (using conda)
- [x] Create the initial file structure using cookiecutter
- [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and 
- [x] Add a model file and a training script and get that running
- [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [x] Remember to comply with good coding practices (`pep8`) while doing the project
- [x] Do a bit of code typing and remember to document essential parts of your code
- [x] Setup version control for your data or part of your data
- [x] Construct one or multiple docker files for your code
- [x] Build the docker files locally and make sure they work as intended
- [x] Write one or multiple configurations files for your experiments
- [x] Used Hydra to load the configurations and manage your hyperparameters
- [x] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [x] Use wandb to log training progress and other important metrics/artifacts in your code
- ~~[ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code~~

### Week 2

- [x] Write unit tests related to the data part of your code
- [x] Write unit tests related to model construction
- [x] Calculate the coverage.
- [x] Get some continues integration running on the github repository
- [x] (optional) Create a new project on `gcp` and invite all group members to it
- [x] Create a data storage on `gcp` for you data
- [x] Create a trigger workflow for automatically building your docker images
- [x] Get your model training on `gcp`
- ~~[ ] Play around with distributed data loading~~
- [x] (optional) Play around with distributed model training
- [ ] Play around with quantization and compilation for you trained models

### Week 3

- [x] ~~Deployed your model locally using TorchServe~~
- [x] Checked how robust your model is towards data drifting
- [x] Deployed your model using `gcp`
- [ ] Monitored the system of your deployed model

### Additional

- [ ] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Make sure all group members have a understanding about all parts of the project
- [ ] Create a presentation explaining your project
- [ ] Uploaded all your code to github
- [ ] (extra) Implemented pre-commit hooks for your project repository
- [ ] (extra) Used Optuna to run hyperparameter optimization on your model
