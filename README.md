# Disease Classification using K-Nearest Neighbors (KNN)

This project applies the K-Nearest Neighbors (KNN) algorithm to classify diseases based on symptoms. The classification is performed using a dataset of symptoms and diseases, with preventive recommendations for each disease.

## Project Structure

- **data/DiseaseAndSymptoms.csv** - Dataset containing diseases and their related symptoms.
- **data/DiseasePrecaution.csv** - Dataset providing preventive recommendations for each disease.
- **knn_disease_classifier.py** - Python script implementing the KNN algorithm for disease classification.

https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset/data - Link to Dataset

## How to Run

1. Clone this repository:

    ```bash
    git clone https://github.com/username/Disease-Classification-KNN.git
    cd Disease-Classification-KNN
    ```

2. Install required libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the classification script:

    ```bash
    python knn_disease_classifier.py
    ```

## KNN Algorithm Overview

The K-Nearest Neighbors (KNN) algorithm is used for classification and regression tasks. This algorithm classifies new observations based on the `k` closest observations in the training data. For disease classification, the algorithm identifies the closest matches based on symptoms and assigns a disease class according to the majority class of its neighbors.

### Dataset

- **DiseaseAndSymptoms.csv**: Contains disease names and associated symptoms, enabling binary feature extraction for each symptom.
- **DiseasePrecaution.csv**: Contains preventive recommendations for each disease to reduce risk or assist in faster recovery.

### Results

This project demonstrates the KNN algorithm’s capability to achieve high accuracy in disease classification by finding the optimal `k` value. The model achieved an accuracy of 100% for certain `k` values (1–8), showing its efficiency in identifying diseases based on symptoms.

## Dependencies

- Python 3.7 or above
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the dependencies by running:

```bash
pip install -r requirements.txt
