export PROJECT_ID=mlops-project-338109
export BUCKET_NAME=${PROJECT_ID}-aiplatform
export IMAGE_REPO_NAME=github.com
export IMAGE_TAG=gpu_latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
export REGION=us-central1
export JOB_NAME=custom_container_job_gpu_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
--scale-tier BASIC_GPU \
--region $REGION \
--master-image-uri $IMAGE_URI \
-- \
gcloud_training=True \
gcloud_dir=$BUCKET_NAME \
n_epochs=2 \
batch_size=1