# Base image
FROM huggingface/transformers-pytorch-cpu

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \ 
    apt clean && rm -rf /var/lib/apt/lists/*


# Copy files
WORKDIR /
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
# RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install -r /requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "-u", "src/models/predict_model.py"]

