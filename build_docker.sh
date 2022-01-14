
docker build -f dockerfiles/train.dockerfile . -t trainer:latest
docker build -f dockerfiles/predict.dockerfile . -t predicter:latest