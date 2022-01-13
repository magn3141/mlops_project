FROM python:3.9-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* \ curl

# Copy files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
WORKDIR /
ENV RUSTUP_HOME=/rust
ENV CARGO_HOME=/cargo 
ENV PATH=/cargo/bin:/rust/bin:$PATH
RUN echo "(curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly --no-modify-path) && rustup default nightly" > /install-rust.sh && chmod 755 /install-rust.sh

RUN pip install -r requirements.txt --no-cache-dir
ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]

