# 📈 Tesla Stock Price Prediction — Time Series Forecasting with LSTM

## 📌 Overview

Forecasting stock prices is a complex and high-impact task in financial analytics due to the inherent volatility and noise in market data. This project leverages **Long Short-Term Memory (LSTM)** neural networks, a type of recurrent neural network (RNN) well-suited for sequential data, to model and predict the closing price of Tesla stock.

The dataset comprises historical stock metrics such as ```Open```, ```High```, ```Low```, ```Close```, and ```Volume```, recorded on a daily basis. The objective is to develop a predictive model that captures the temporal dependencies in stock behaviour to forecast future closing prices.

---

## 🎯 Project Objective

Develop a deep learning model using LSTM architecture to predict Tesla’s stock **closing price** by learning from historical patterns in time series data.

---

## 🛠️ Project Workflow

| Step | Description |
|------|-------------|
| 1. Data Acquisition | Loaded daily Tesla stock data with features including ```Open```, ```High```, ```Low```, ```Close```, and ```Volume```. |
| 2. Data Preprocessing | - Normalized values using `MinMaxScaler`<br>- Generated input sequences with sliding window approach (window size = 14)<br>- Split data into training and testing sets |
| 3. Model Architecture | Built a deep LSTM model with:<br>→ Two stacked `LSTM` layers<br>→ One fully connected `Dense` layer<br>→ `Dropout` layer for regularization |
| 4. Model Training | Compiled with the `Adam` optimizer and `mean_squared_error` loss; used `EarlyStopping` to monitor validation loss and avoid overfitting. |
| 5. Prediction & Inversion | Predicted scaled values and inverse-transformed them using the scaler to obtain actual stock price predictions. |
| 6. Evaluation | Evaluated model using `MAE` and `RMSE` to quantify prediction accuracy. |
| 7. Visualization | Plotted actual vs predicted stock prices to assess temporal alignment. |

---

## 🧠 Model Architecture Summary
Layer (type)      Output Shape    Param #  
=========================================
LSTM (128)        (None, 14,128)   66,560  
LSTM (128)        (None,128)      131,584  
Dense (64, relu)  (None,64)         8,256  
Dropout (0.2)     (None,64)             0  
Dense (1, tanh)   (None,1)             65  
=========================================
Total params: 206,465

---

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Mean Absolute Error (`MAE`) | 5.4085 |
| Root Mean Squared Error (`RMSE`) | 7.2318 |

---

## 📈 Visualization — Actual vs Predicted

The model’s predictions closely align with the actual closing prices, indicating that the LSTM layers effectively captured temporal dynamics in the dataset.



---

## 💡 Key Insights

- **Temporal Modeling**: LSTM networks are highly effective for learning sequential dependencies in time-series data such as stock prices.
- **Data Preparation**: Proper normalization and sequence generation were critical for model performance.
- **Overfitting Control**: `EarlyStopping` callback based on validation loss helped reduce overfitting and preserved model generalization.
- **Real-World Challenges**: Missing dates (e.g., weekends, holidays) are typical in stock market data and must be accounted for in the modeling process.

---

## 📖 Notes

- **Dataset Source**: [Tesla Stock Price Dataset – Kaggle](https://www.kaggle.com/datasets/jillanisofttech/tesla-stock-price?select=Tasla_Stock_Updated_V2.csv)
- **Libraries Used**: `TensorFlow/Keras`, `NumPy`, `Pandas`, `scikit-learn`, `Matplotlib`, `Seaborn`
- **Input Window Size**: 14 days
- **Target Variable**: `Close` price

---

## 🚀 Potential Improvements

- Integrate additional technical indicators (e.g., moving averages, RSI, MACD)
- Incorporate multi-variate signals (e.g., macroeconomic indicators, news sentiment)
- Explore alternative architectures such as `GRU`, `Transformer`, or hybrid CNN–LSTM models
- Deploy model as a real-time prediction app using `Streamlit` or `Flask`

---


