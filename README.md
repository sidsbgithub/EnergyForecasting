# Energy Load Forecasting with Spatio-Temporal Graph Neural Networks

This project develops and evaluates a deep learning model to forecast hourly energy load across multiple interconnected power zones. It leverages a Spatio-Temporal Graph Neural Network (STGNN) built with PyTorch and PyTorch Geometric to capture both the time-series nature of energy consumption and the spatial relationships between different geographical zones. The model uses historical load data and corresponding weather data as input features. The project includes notebooks for data processing, feature engineering, model training, and hyperparameter optimization using Optuna.

---

## Features

- **Data Processing:** Robust pipelines for cleaning, aligning, and processing PJM load data and Open-Meteo weather data.
- **Feature Engineering:** Creation of cyclical time-based features and lagged variables to improve model accuracy.
- **Advanced Modeling:** Implementation of an STGNN architecture combining a Graph Convolutional Network (GCN) for spatial dependencies and an LSTM for temporal patterns.
- **Hyperparameter Tuning:** Automated search for optimal model parameters using Optuna, with results stored for reproducibility.
- **Evaluation:** Comprehensive model evaluation on a held-out test set using metrics like RMSE, MAE, R2, and an adjusted MAPE.

---

## Project Structure

```
├── data/
│   ├── pjm_load_2022_2025/
│   ├── processed/
│   └── raw_weather_data/
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_Load_Data_Processing.ipynb
│   ├── 03_Weather_Data_Processing.ipynb
│   ├── 04_Data_Combination_and_Preprocessing.ipynb
│   ├── 05_Model_Building_and_Training.ipynb
│   └── 06-error-analysis.ipynb
├── src/
│   └── model_ready/
├── .gitattributes
├── .gitignore
├── README.md
├── config.py
└── requirements.txt
```

---

## Workflow

### 1. Introduction

The `01_introduction.ipynb` notebook provides a basic introduction to the project and demonstrates fundamental data manipulation with pandas.

### 2. Load Data Processing

The `02_Load_Data_Processing.ipynb` notebook is responsible for processing the raw PJM load data. It reads the data for different zones and years, combines them, and performs necessary cleaning and preprocessing steps.

### 3. Weather Data Processing

The `03_Weather_Data_Processing.ipynb` notebook handles the processing of raw weather data from Open-Meteo. It processes the data for each zone and combines them into a single file.

### 4. Data Combination and Preprocessing

The `04_Data_Combination_and_Preprocessing.ipynb` notebook combines the processed load and weather data. It then performs feature engineering, creating cyclical time-based features and lagged variables, and splits the data into training, validation, and test sets.

### 5. Model Building and Training

The `05_Model_Building_and_Training.ipynb` notebook is where the STGNN model is built and trained. This notebook uses Optuna for hyperparameter tuning to find the best model parameters. The best model is then saved for later use.

### 6. Error Analysis

The `06-error-analysis.ipynb` notebook is used for a detailed analysis of the trained model's performance. It loads the saved model and evaluates it on the test set, providing various error metrics and visualizations.

---

## Dependencies

The project's dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

The key libraries used are:

- pandas
- numpy
- scikit-learn
- joblib
- torch
- torch_geometric
- optuna
- matplotlib
- plotly

---

## Model Architecture

The model is a Spatio-Temporal Graph Neural Network (STGNN) that combines a Graph Convolutional Network (GCN) and a Long Short-Term Memory (LSTM) network.

- **GCN:** The GCN component is used to learn the spatial dependencies between the different power zones. It takes the graph structure of the power grid as input and learns representations of the zones that incorporate information from their neighbors.
- **LSTM:** The LSTM component is used to model the temporal dependencies in the time-series data. It takes the output of the GCN at each time step and learns to predict the future energy load.

---

## Model Performance and Evaluation 📊

The model's performance was rigorously evaluated on a held-out test set. The key performance metrics are summarized below:

| Metric | Value |
| --------------- | -------- |
| **RMSE** | 227.15 |
| **MAE** | 159.29 |
| **R² Score** | 0.98 |
| **Adjusted MAPE**| 2.94% |

### Observations and Insights

* The **R² score of 0.98** indicates that the model can explain 98% of the variance in the test data, which is an excellent fit.
* The **Adjusted MAPE of 2.94%** shows that the model's predictions are, on average, within 3% of the actual values. This is a very strong result for energy forecasting.
* **Error analysis** reveals that the largest errors tend to occur during extreme weather events or holidays, which are notoriously difficult to predict.
* **Zone-level performance** is consistent across all zones, with no single zone showing significantly worse performance than the others. This demonstrates the model's ability to generalize well across different geographical areas.

---

## Usage

To use the trained model for prediction, you can use the `06-error-analysis.ipynb` notebook as a reference. The general steps are:

1. Load the trained model from the `src/model_ready/` directory.
2. Load the feature scaler from the `src/model_ready/` directory.
3. Prepare the input data in the same format as the training data.
4. Use the model to make predictions on the new data.
5. Inverse transform the predictions to get the actual energy load values.

---

## Contributing

Contributions to this project are welcome. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a clear commit message.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.