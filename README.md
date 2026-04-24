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

Open `index.html` with any browser.
View backend documentation through `http://localhost:8000/docs`.

## TODOs:

#### dataset.ipynb
- Add class balance analysis.

#### eda.ipynb
- Add more diverse visualizations.
- Write down insights/observations.

#### train.ipynb
- Fix all broken code, refactor and rerun.
- Extract constants and params into a config YAML file or just a separate reference notebook (maybe).

#### test.ipynb
- Fix all broken code, refactor and rerun.

#### Deployment with Docker
- Try running the Dockerfile on different machines.
- Ensure there aren't any big errors and that we're not missing anything.

#### Presentation
- Test Script - Automated API testing (`test_api.py`).
- 10min demo with live API testing.
- Presentation slides?

### Required Deliverables
- ~~1. EDA Notebook - 8+ visualizations~~ with written insights.
- ~~2. Trained Model - Exported as .pth file.~~
- 3. Training Report - Loss/accuracy curves, best epoch, final metrics.
- 4. Evaluation - Confusion matrix, per-class accuracy, F1 scores.
- ~~5. FastAPI App - /predict endpoint accepting image uploads.~~
- ~~6. Dockerfile - Containerized deployment ready.~~
- 7. Test Script - Automated API testing (`test_api.py`).
- 8. Presentation - 10-minute demo with live API test.

### Questions to the TA
- What should `test_api.py` be?
- Ask whether we need to train DenseNet as well and compare the results or is a single model sufficient.
- Are we missing anything in the deployment part (Should we ship the frontend on Docker as well?).

