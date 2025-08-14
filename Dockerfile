FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder
RUN apt-get -y update && apt-get install -y \
  build-essential cmake git pkg-config \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libgtk-3-dev libcanberra-gtk3-module \
  python3-dev python3-numpy
ENV VER=4.12.0
WORKDIR /src
RUN git clone https://github.com/opencv/opencv -b $VER && \
  git clone https://github.com/opencv/opencv_contrib -b $VER
WORKDIR /src/opencv/build
RUN cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D WITH_CUDA=ON -D WITH_CUBLAS=1 ..
RUN make -j && make install && ldconfig -v
RUN python3 -c 'import cv2 ; print(cv2.cuda.getCudaEnabledDeviceCount())'

FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
COPY --from=builder /usr/local /usr/local/
RUN apt-get -y update && apt-get install -y \
  python3-dev python3-numpy python3-pip \
  libharfbuzz0b libgtk-3-0t64 libavcodec60 \
  libavformat60 libswscale7 libwebpdemux2
WORKDIR /root
COPY run_rand.py run_rand.py
ENTRYPOINT ["./run_rand.py"]
# docker build -f Dockerfile . -t testit && time docker run --gpus=all -ti testit
