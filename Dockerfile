FROM registry.parqour.com/cv/dist/base-builder:v1.0.0-base-trt8
ENV CUDA_MODULE_LOADING=LAZY
RUN mkdir -p /app
WORKDIR /app


### Parking Layer
COPY models/ models/
COPY src/ src/
COPY CMakeLists.txt .
RUN mkdir build && \
    cd build && \
    cmake -D CMAKE_C_COMPILER=gcc-9 -D CMAKE_CXX_COMPILER=g++-9 .. && \
    make -Werror -Wall && \
    mv cpp-app ../ && \
    cd .. && \
    rm -rf src/ build/ CMakeLists.txt
RUN echo "build prepared"
