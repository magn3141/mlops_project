from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, jsonify, request
from google.cloud import storage
import logging
import hydra
from omegaconf import DictConfig


app = Flask(__name__)
log = logging.getLogger(__name__)
tokenizer = None
model = None


@hydra.main(config_path="config", config_name="serve")
def main(cfg: DictConfig):
    working_dir = hydra.utils.get_original_cwd()
    bucket = cfg.bucket
    gcloud_dir = cfg.cloud_dir
    local_dir = working_dir + cfg.local_dir

    log.info(
        f"Downloading files from bucket: {bucket} in directory: {local_dir}")

    if (cfg.run_locally is True):
        key = cfg.profile_file_path
        storage_client = storage.Client.from_service_account_json(
            working_dir + key)
    else:
        storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    blobs = bucket.list_blobs(prefix=gcloud_dir)
    log.info(f"Saving in local directory: {local_dir}")
    for blob in blobs:
        filename = blob.name.replace(gcloud_dir, "")
        blob.chunk_size = 5 * 1024 * 1024
        blob.download_to_filename(local_dir + filename)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.backbone, model_max_length=cfg.tokenizer_max_length)
    global model
    model = AutoModelForCausalLM.from_pretrained(local_dir)


@app.route('/generate-text', methods=['POST'])
def generate_text():
    request_json = request.get_json()
    if request_json and 'message' in request_json and 'max_length' in request_json:
        message = request_json['message']
        max_length = request_json['max_length']
        token = tokenizer.encode(message, return_tensors='pt')
        generated = model.generate(token, max_length=int(max_length))[0]
        decoded = tokenizer.decode(generated)
        return decoded
    else:
        return 'Input needs to be a json object containing a message and a max_length attribute'


if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port=8080)
