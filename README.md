# Deep Learning Midterm Project

Two deep learning models built with TensorFlow/Keras:

1. **DNN** - predicts whether a telecom customer will churn (leave the company) from their account and billing data.
2. **CNN** - classifies chest X-ray images as **Normal** or **Pneumonia**.

The full write-up (problem definition, preprocessing, architecture, results and conclusions) is in `Project_Report.pdf`.

## Results

| Model | Dataset | Test accuracy | ROC-AUC | Notes |
|-------|---------|---------------|---------|-------|
| DNN | Telco Customer Churn | 79.7% | 0.837 | Churn recall 48%, no-churn recall 91% |
| CNN | Chest X-Ray Images (Pneumonia) | 71.6% | 0.915 | Pneumonia recall 99.7%, normal recall 25% |

## Files

| File | Description |
|------|-------------|
| `churn_dnn.ipynb` | DNN for customer churn prediction |
| `pneumonia_cnn.ipynb` | CNN for pneumonia detection |
| `best_cnn.keras` | Trained CNN (best epoch, saved by ModelCheckpoint) |
| `Project_Report.pdf` | Project report |
| `requirements.txt` | Python libraries needed |

## Datasets

The datasets are not included in this repo because of their size. Download them from Kaggle:

- Telco Customer Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Chest X-Ray Images (Pneumonia): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia (about 2.3 GB)

## How to run

1. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Download and extract the datasets.
3. Change the dataset paths in the notebooks to where you saved them:
   - `churn_dnn.ipynb`, section 1: the path in `pd.read_csv(...)`
   - `pneumonia_cnn.ipynb`, section 1: `base_dir` (the `chest_xray` folder that has `train`, `val` and `test` inside)
4. Open the notebooks in Jupyter and run all cells.

The churn notebook trains in about a minute. The CNN took about 15 minutes per epoch on a laptop CPU (8 epochs, around 2 hours), so a GPU (e.g. Kaggle or Google Colab) is recommended. To test the CNN without retraining, skip the training cell and load `best_cnn.keras` (the notebook already has a cell for this).

## Tools

Python 3.12, TensorFlow 2.21, scikit-learn, pandas, NumPy, Matplotlib, Seaborn.
