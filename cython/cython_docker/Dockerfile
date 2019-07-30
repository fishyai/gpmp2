FROM ubuntu:16.04

RUN apt update -y
RUN apt install -y --no-install-recommends \
  cmake ca-certificates \
  libboost-all-dev \
  git \
  python \
  vim \
  apt-utils \
  python-pip \
  wget \
  build-essential

RUN update-ca-certificates
RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install ipython numpy matplotlib scipy

RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0.tar.gz && tar -xvf cmake-3.15.0.tar.gz
WORKDIR /cmake-3.15.0
RUN cmake . && make && make install
RUN cmake --version

WORKDIR /
RUN git clone https://github.com/borglab/gtsam.git && pwd

WORKDIR /gtsam
RUN pip install -r cython/requirements.txt && mkdir build
WORKDIR /gtsam/build
RUN cmake \
  -DGTSAM_BUILD_UNSTABLE=OFF \
  -DGTSAM_INSTALL_CYTHON_TOOLBOX=ON ..
RUN pwd && make -j5 && make install

ENV PYTHONPATH $PYTHONPATH:/usr/local/cython/

WORKDIR /
RUN git clone https://github.com/fishyai/gpmp2.git && cd gpmp2

WORKDIR /gpmp2
RUN git checkout origin/cython && git checkout -b cython
RUN mkdir build

WORKDIR /gpmp2/build
RUN cmake \
  -DGPMP2_BUILD_CYTHON_TOOLBOX=ON ..
RUN make check -j5
RUN make install

# The following is left in because maybe it'll help enable matplotlib in Docker
# TODO figure out how to actually enable matplotlib in docker

# RUN echo "tzdata tzdata/Zones/America select Los Angeles" | debconf-set-selections && apt install python-tk -y
#
# RUN apt install sudo -y
#
# RUN export uid=1000 gid=1000 && \
#     mkdir -p /home/developer && \
#     echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
#     echo "developer:x:${uid}:" >> /etc/group && \
#     echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
#     chmod 0440 /etc/sudoers.d/developer && \
#     chown ${uid}:${gid} -R /home/developer
#
# USER developer
# ENV HOME /home/developer

WORKDIR /gpmp2/cython
