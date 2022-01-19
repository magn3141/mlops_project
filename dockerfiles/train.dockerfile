# Base image
FROM python:3.9.9-slim

# install required commands through apt
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Download rust compiler to fix transformers and add it to path
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ARG wandb_api_key
ENV WANDB_API_KEY=$wandb_api_key

# Copy files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
WORKDIR /
RUN mkdir models
# Install python dependencies 
RUN pip install -U pip
RUN pip install -r requirements.txt --no-cache-dir


ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

