# Base image
FROM python:3.9.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
WORKDIR /
RUN pip install -U git+https://github.com/huggingface/transformers.git@v4.13.0
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-V"]

