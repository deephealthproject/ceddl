FROM alpine:3.13

# Install prerequisites
RUN apk update && \ 
    apk add build-base=0.5-r2 && \
    apk add bash=5.1.0-r0 && \
    apk add libexecinfo-dev=1.1-r1 && \
    apk add git && \
    apk add cmake && \
    apk add zlib-dev && \
    rm -rf /var/cache/apk/*  

# Build the eddl.so from the develop branch
RUN echo "START GIT CLONE OF EDDL" && \
    git clone --recurse-submodules -j8 https://github.com/deephealthproject/eddl.git && \
    cd eddl && \
    # checkout the specific version the ceddl will be built against (26-Mar-2021)
    git checkout 77451e18cbfcab02706cb7156a27b6e53ece5aa6 && \
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