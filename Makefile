image=doc.smartparking.kz/parking_cpp
image_infra=registry.infra.smartparking.kz/parking_cpp
tag=v6.7.0

build:
	docker build -t ${image}:${tag} -f dockerfiles/DockerfileParking .

build_infra:
	docker build -t ${image_infra}:${tag} -f Dockerfile .

push:
	docker push ${image}:${tag}

push_infra:
	docker push ${image_infra}:${tag}

create_app:
	make build_infra
	make push_infra

run:
	docker-compose -f parking_cpp.yml up -d

restart:
	docker-compose -f parking_cpp.yml down && docker-compose -f parking_cpp.yml up -d

stop:
	docker-compose -f parking_cpp.yml down

prod:
	rm -f ./prod && mkdir ./prod && cp -r src/ prod/ \
	&& cp -r models/ prod/ && cp -r Dockerfile prod/ \
	&& cp CMakeLists.txt prod/ && cp config.json prod/ \
	&& cp parking_cpp.yml prod/ && cp Makefile prod/ \
	&& echo "Production prepared"

run_dev:
	xhost local:root && xhost +  && docker run -d -it --rm --cap-add sys_ptrace -p127.0.0.1:2222:22 \
				--gpus all -e DISPLAY="$DISPLAY" \
				-v /tmp/.X11-unix:/tmp/.X11-unix \
				--name parking_cpp_local \
				registry.infra.smartparking.kz/parking_cpp_dev:1.2
