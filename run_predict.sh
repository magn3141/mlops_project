name="$1"
shift;
# docker run --name predicter_1 -v %cd%/models/running_model/:/models/running_model/ predicter:latest text="Velkommen: " relative_path="models/running_model"
docker run --name "$name"  -v "$(pwd)"/models:/models -v "$(pwd)"/outputs:/outputs predicter:latest $@