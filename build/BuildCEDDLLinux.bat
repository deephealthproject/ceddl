pushd %~dp0\.\..\

if not exist bin mkdir output

:: build image
docker build . -f.\src\Dockerfile --tag philips/buildceddl --build-arg http_proxy=%http_proxy% --build-arg https_proxy=%https_proxy%

:: Create container from the image with name 'buildceddlcontainer' and let it run with 
:: bash interactive terminal to keep it from closing (-it -d)
docker run -it -d --name buildceddlcontainer --rm philips/buildceddl bash

:: Copy shared libraries from the container
docker cp buildceddlcontainer:/eddl/build/lib64/libeddl.so ./output
docker cp buildceddlcontainer:/ceddl/src/libceddl.so ./output

:: kill container.
docker kill buildceddlcontainer
docker rmi -f philips/buildceddl

popd
