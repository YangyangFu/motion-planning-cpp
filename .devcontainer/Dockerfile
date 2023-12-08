FROM ubuntu:20.04 
LABEL maintainer="yangyangfu <fuyy2008@gmail.com>" \
      description="Docker image for building and testing the C++ implementation of the Open Neural Network Exchange (ONNX) models" \
      version="1.0"

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdb \
    cmake \
    git \
    libcairo2-dev \
    libgraphicsmagick1-dev \
    libpng-dev \
    wget \
    ca-certificates \
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
USER root
RUN cd $HOME \
    && git clone --recurse-submodules https://github.com/cpp-io2d/P0267_RefImpl.git \
    && cd P0267_RefImpl \
    && mkdir build \
    && cd build \
    && cmake --config Release "-DCMAKE_BUILD_TYPE=Release" .. \
    && make \
    && make install 

# mount volume
RUN cd $HOME \
    && mkdir -p /home/developer/cpp

# start 
USER developer
WORKDIR $HOME