FROM hsdp-cp-docker.artifactory-ehv.ta.philips.com/application-service-base:1.0.0 AS build-EDDL

# Install prerequisites
RUN apt-get update && \
    apt-get install -y \
    git \
    cmake \
    build-essential \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build the eddl.so from the develop branch
RUN echo "START GIT CLONE OF EDDL" && \
    git clone --recurse-submodules -j8 https://github.com/deephealthproject/eddl.git && \
    cd eddl && \
    # checkout the specific version the ceddl will be built against (1-Mar-2021)
    git checkout 676c91823e9483589fc05e51a6191755b27a0e49 && \
    mkdir build && \
    cd build && \
    echo "START CMAKE EDDL" && \
    # Set CMAKE_BUILD_TYPE=Debug if you want debug library (also adjust makefile.ansi accordingly).
    cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TARGET=CPU -DBUILD_SHARED_LIBS=ON -DBUILD_SUPERBUILD=ON && \
    echo "START MAKE OF EDDL" && \
    make

# Build the ceddl.so
RUN mkdir -p /ceddl/src
COPY ./src/ceddl.c /ceddl/src
COPY ./src/ceddl.h /ceddl/src
COPY ./src/makefile.ansi /ceddl/src

RUN echo "START BUILD OF CEDDL" && \
    cd /ceddl/src && \
    make -f makefile.ansi