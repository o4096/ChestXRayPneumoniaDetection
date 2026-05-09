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

## TODOs:

#### eda.ipynb
- Add more diverse visualizations.
- Write down insights/observations.

#### train.ipynb
- Extract constants and params into a config YAML file or just a separate reference notebook (maybe).

#### test.ipynb
- Separate the code from train.ipynb
- Fix the recall threshold tuning and use it in deployment.

### Required Deliverables
- 1. EDA Notebook - 8+ visualizations with written insights.
~~- 2. Trained Model - Exported as .pth file.~~
~~- 3. Training Report - Loss/accuracy curves, best epoch, final metrics.~~
~~- 4. Evaluation - Confusion matrix, Precision/Recall/F1, ROC-AUC, threshold tuning for Recall.~~
~~- 5. FastAPI App - /predict endpoint accepting image uploads.~~
~~- 6. Dockerfile - Containerized deployment ready.~~
~~- 7. Test Script - Automated API testing (`test_api.py`).~~
~~- 8. Presentation - 10-minute demo with live API test.~~

