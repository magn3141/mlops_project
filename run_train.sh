name = $1;
shift;
docker run --name $name  -v %cd%/data:/data trainer:latest $@