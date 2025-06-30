# Energy Load Forecasting with Spatio-Temporal Graph Neural Networks

This project develops and evaluates a deep learning model to forecast hourly energy load across multiple interconnected power zones. It leverages a Spatio-Temporal Graph Neural Network (STGNN) built with PyTorch and PyTorch Geometric to capture both the time-series nature of energy consumption and the spatial relationships between different geographical zones.

The model uses historical load data and corresponding weather data as input features. The project includes notebooks for data processing, feature engineering, model training, and hyperparameter optimization using Optuna.


## Features

- **Data Processing:** Robust pipelines for cleaning, aligning, and processing PJM load data and Open-Meteo weather data.
- **Feature Engineering:** Creation of cyclical time-based features and lagged variables to improve model accuracy.
- **Advanced Modeling:** Implementation of an STGNN architecture combining a Graph Convolutional Network (GCN) for spatial dependencies and an LSTM for temporal patterns.
- **Hyperparameter Tuning:** Automated search for optimal model parameters using Optuna, with results stored for reproducibility.
- **Evaluation:** Comprehensive model evaluation on a held-out test set using metrics like RMSE, MAE, R2, and an adjusted MAPE.
- **Deployment-Ready:** Includes a `predict.py` script designed for operational forecasting, demonstrating how to use the trained model on new data.


