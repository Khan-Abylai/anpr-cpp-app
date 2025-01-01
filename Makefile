image_base=registry.infra.smartparking.kz/cpp-base:1.0
image=registry.parqour.com/cv/application/anpr-cpp-app
tag=v1.0.0-malaysia
dev_tag=dev
engine_builder_tag=${tag}-engine
container_name=test
build_base:
	docker build -t ${image_base} -f dockerfiles/Dockerfile-base .

build_dev:
	docker build -t ${image}:${dev_tag} -f dockerfiles/Dockerfile-dev .

run_dev:
	docker run -d -it --rm --cap-add sys_ptrace -p127.0.0.1:2222:22 \
            --gpus all -e DISPLAY="$DISPLAY" \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name parking_cpp_local \
            ${image}:${dev_tag}

run_dev_2:
	docker run -d -it --rm --cap-add sys_ptrace -p0.0.0.0:2222:22 \
		--gpus all --name parking_cpp_remote ${image}:${dev_tag}

build_app:
	docker build -t ${image}:${tag} --build-arg CUDA_MODULE_LOADING="LAZY" .

stop_dev:
	docker stop parking_cpp_local

build_engine_builder:
	docker build -t ${image}:${engine_builder_tag} --build-arg CUDA_MODULE_LOADING="LAZY" .

push_app:
	docker push ${image}:${tag}

prod:
	rm -f ./prod && mkdir ./prod && cp -r src/ prod/ \
	&& cp -r models/ prod/ && cp -r Dockerfile prod/ \
	&& cp CMakeLists.txt prod/ && cp config.json prod/ \
	&& cp parking_app.yml prod/ && cp Makefile prod/ \
	&& cp -r engine_builder prod/ && echo "Production prepared"

end2end:
	make build_app
	make push_app