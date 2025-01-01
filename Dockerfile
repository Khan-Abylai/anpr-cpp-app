FROM registry.parqour.com/cv/dist/base-builder:parking_cpp_base_1.0
RUN mkdir -p /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Almaty /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY models models
COPY src/ src/
COPY CMakeLists.txt .

RUN mkdir build && \
    cd build && \
    cmake -j 8  -DCMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=gcc-9 -D CMAKE_CXX_COMPILER=g++-9 .. && \
    make -j 8 && \
    mv parking ../ && \
    cd .. && \
    rm -rf src/ build/ CMakeLists.txt
