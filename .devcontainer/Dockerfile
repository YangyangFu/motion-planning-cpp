FROM nvidia/cuda:12.3.0-devel-ubuntu20.04
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
    dirmngr \
    gnupg2 \
    gdb \ 
    && rm -rf /var/lib/apt/lists/*

# set up ROS noetic keys
RUN set -eux; \
       key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'; \
       export GNUPGHOME="$(mktemp -d)"; \
       gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
       mkdir -p /usr/share/keyrings; \
       gpg --batch --export "$key" > /usr/share/keyrings/ros1-latest-archive-keyring.gpg; \
       gpgconf --kill all; \
       rm -rf "$GNUPGHOME"

# setup sources.list
RUN echo "deb [ signed-by=/usr/share/keyrings/ros1-latest-archive-keyring.gpg ] http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup ROS environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies if needed
RUN apt-get update --fix-missing\
    && apt-get -y --no-install-recommends install \
    git \
    gcc \
    vim \
    psmisc \
    libxml2-dev \
    libxslt-dev \
    python3 \
    python3-pip \ 
    python-is-python3\
    ros-noetic-amcl \
    ros-noetic-base-local-planner \
    ros-noetic-map-server \
    ros-noetic-move-base \
    ros-noetic-navfn \
    && rm -rf /var/lib/apt/lists/*

# python packages
RUN pip3 install setuptools && pip3 install catkin-tools && pip3 install conan==1.59.0

# bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
