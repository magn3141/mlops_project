# Base image
# FROM python:3.9.9-slim
FROM --platform=linux/amd64 python:3.9.9-slim


# install required commands through apt
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git curl wget && \
    apt clean && rm -rf /var/lib/apt/lists/* 

# Download rust compiler to fix transformers and add it to path
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /root

# Copy files
COPY requirements.txt /root/requirements.txt
COPY setup.py /root/setup.py
COPY src/ /root/src/
COPY data/ /root/data/
RUN mkdir /root/models
RUN pip install -U pip
RUN pip install -r /root/requirements.txt --no-cache-dir

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

ENV WANDB_API_KEY=""
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

