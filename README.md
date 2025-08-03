# Time Series Forecasting

This repository is a collection of Jupyter notebooks that explore and implement various techniques for time series forecasting. The project provides a comparative study of different forecasting models, including traditional machine learning and state-of-the-art deep learning architectures. The core of this work is to showcase the practical application of these models using a Forex dataset.

## Project Goal

The main objective of this project is to serve as a comprehensive resource for implementing diverse time series forecasting models. Each notebook is a self-contained example, guiding users through data preprocessing, model training, and evaluation. This allows practitioners to understand the strengths and weaknesses of different approaches and select the most suitable model for their specific forecasting needs.

## Notebooks

### `FORECASTING_WITH_GBM_&_LIGHTGBM_&_CAT.ipynb`

This notebook focuses on using powerful Gradient Boosting Machines (GBMs) for forecasting. It includes a detailed workflow for data preparation and model training.

**Models Implemented:**

| Model | Description |
| :--- | :--- |
| **LightGBM** | A fast, distributed, high-performance gradient boosting framework based on decision tree algorithms. |
| **CatBoost** | A gradient boosting library with categorical feature support. It is known for its high accuracy and robust handling of various data types. |

**Key Features & Steps:**

  * **Data Loading**: Loads a Forex dataset from a CSV file. The data includes `DateTime Stamp`, `Bar OPEN Bid Quote`, `Bar HIGH Bid Quote`, `Bar LOW Bid Quote`, `Bar CLOSE Bid Quote`, and `Volume` columns.
  * **Feature Engineering**: Creates new features from the `DateTime Stamp` and other columns to capture temporal patterns.
  * **Data Splitting**: Uses a time-based train-test split to simulate a real-world forecasting scenario.
  * **Model Training & Evaluation**: Trains LightGBM and CatBoost models and evaluates their performance using `mean_squared_error` from `scikit-learn`.

**Example Data Snippet:**
A sample of the data loaded in the notebook shows the structure of the Forex dataset.

| DateTime Stamp | Bar OPEN Bid Quote | Bar HIGH Bid Quote | Bar LOW Bid Quote | Bar CLOSE Bid Quote | Volume |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 20070930 170000 | 0.8832 | 0.8832 | 0.8832 | 0.8832 | 0 |
| 20071007 170000 | 0.8823 | 0.8823 | 0.8823 | 0.8823 | 0 |
| 20071007 170100 | 0.8822 | 0.8822 | 0.8822 | 0.8822 | 0 |
| 20071007 170200 | 0.8822 | 0.8823 | 0.8822 | 0.8822 | 0 |

### `FORECASTING_USING_TFT.ipynb`

This notebook demonstrates the implementation of the Temporal Fusion Transformer (TFT) model, a deep learning architecture designed for interpretable multi-horizon forecasting.

**Models Implemented:**

| Model | Description |
| :--- | :--- |
| **Temporal Fusion Transformer (TFT)** | A deep learning model that combines a multi-head attention mechanism with recurrent layers to enable high-performance forecasting on complex time series data while providing interpretable outputs.

**Key Features & Steps:**

  * **Data Handling**: Uses the `pytorch-forecasting` library for efficient data handling and preprocessing, including a `TimeSeriesDataSet` class to prepare data for the TFT model.
  * **Model Training**: The model is trained using the `PyTorch Lightning` framework. The training process is monitored with metrics such as `train_loss` and `val_loss`.
  * **Hyperparameters**: The notebook uses a fixed prediction length and encoder length, and a specific hidden size for the TFT model.

### `FORECASTING_USING_TCN_and_WAVENET.ipynb`

This notebook explores convolutional neural networks for time series forecasting, specifically Temporal Convolutional Networks (TCNs) and WaveNet architectures.

**Models Implemented:**

| Model | Description |
| :--- | :--- |
| **Temporal Convolutional Network (TCN)** | A neural network that uses dilated causal convolutions to process sequences. TCNs are known for their long-range dependency capture and parallel computation capabilities.
| **WaveNet** | An architecture primarily used for generating raw audio, adapted here for forecasting. It uses stacked dilated causal convolutional layers to build a receptive field that can cover a wide range of input data.

**Key Features & Steps:**

  * **PyTorch Implementation**: The models are built and trained using the PyTorch library.
  * **Causal Convolutions**: The notebook demonstrates the use of causal convolutions to ensure that predictions at a given time step only depend on past data.
  * **Model Training**: The models are trained to forecast the `Bar CLOSE Bid Quote` using an optimizer and loss function suitable for regression.

## Dependencies

You can install the necessary dependencies for each notebook using the following commands:

  * **For `FORECASTING_WITH_GBM_&_LIGHTGBM_&_CAT.ipynb`**

    ```bash
    pip install pandas numpy lightgbm catboost scikit-learn
    ```

  * **For `FORECASTING_USING_TFT.ipynb`**

    ```bash
    pip install pytorch-forecasting pytorch-lightning
    ```

  * **For `FORECASTING_USING_TCN_and_WAVENET.ipynb`**

    ```bash
    pip install torch torchvision
    ```

## Data

All notebooks utilize a Forex dataset stored in a Google Drive file named `DAT_ASCII_AUDCAD_M1_2007.csv`. To replicate the results, you must mount your Google Drive within the Colab environment and ensure the file is located at the path `/content/drive/MyDrive/Forex_Data/DAT_ASCII_AUDCAD_M1_2007.csv`.
