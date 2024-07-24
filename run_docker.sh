xhost +local:root
distro=noetic
docker run -it \
     --gpus all \
     --env="DISPLAY" \
     --env="QT_X11_NO_MITSHM=1" \
     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
     motion-planning-cuda:12.3 \
     /bin/bash