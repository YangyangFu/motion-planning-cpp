FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
LABEL maintainer="yangyangfu <fuyy2008@gmail.com>" \
      description="Docker image for building and testing the C++ implementation of motion planning" \
      version="1.0"

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    libgtest-dev \
    libopencv-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# create a user 
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    mkdir -p /etc/sudoers.d && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer && \
    mkdir -m 1777 /tmp/.X11-unix


ENV HOME /home/developer

# Install IO2D to /usr/local/lib
#USER root
#RUN cd $HOME \
#    && git clone --recurse-submodules https://github.com/cpp-io2d/P0267_RefImpl.git \
#    && cd P0267_RefImpl \
#    && mkdir build \
#    && cd build \
#    && cmake --config Release "-DCMAKE_BUILD_TYPE=Release" .. \
#    && make \
#    && make install 

# mount volume
#RUN cd $HOME \
#    && mkdir -p /home/developer/cpp

# start 
USER developer
WORKDIR $HOME