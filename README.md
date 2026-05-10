# ChestXRayPneumoniaDetection
Fine-tuned ResNet18 and DenseNet121 for Pneumonia Detection (Image Classification)

## Build
### Python
```
pip install -r requirements.txt
```
```
python -m uvicorn app:app --reload --port 8000
```

### Docker
build the server container (use `--network=host` to bypass bridge and potentially speed up the build process)
```shell
docker build -t pneumo-detect
```
run the server container
```shell
docker run -d -p 8000:8000 --name pneumo-server pneumo-detect
```
Stop the server container
```shell
docker stop pneumo-server
```
Remove the container
```shell
docker rm pneumo-server
```

## Usage
Access the frontend dashboard through `localhost:8000`.
View backend documentation through `localhost:8000/docs`.

