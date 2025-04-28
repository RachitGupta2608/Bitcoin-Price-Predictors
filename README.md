
# 💰 Bitcoin Price Prediction Project 💰

This project builds a Machine Learning model to predict Bitcoin price movements using Random Forest Classifier and LSTM. It includes the full pipeline of data collection, feature engineering, model training, prediction, and result visualization, all implemented in Python. By analyzing historical data from Yahoo Finance, the model aims to predict whether Bitcoin's price will rise or fall, offering valuable insights into cryptocurrency market trends.

## 🌟 Project Overview

Bitcoin, the world's most popular cryptocurrency, has a volatile price history. In this project, we dive into **predicting Bitcoin price movements** using **two powerful models**:

1. **Random Forest Classifier** 🌳: A robust ensemble model that uses decision trees to predict whether Bitcoin’s price will go up or down.
2. **LSTM (Long Short-Term Memory)** 🧠: A deep learning model perfect for time series data like Bitcoin’s price, capturing past patterns to predict future trends.

Both models are trained using data from **Yahoo Finance**, and feature engineering techniques are applied to extract important market signals, improving prediction accuracy.



## 📊 Data Collection & Exploratory Data Analysis

The dataset used in this project is **Bitcoin's historical price data** obtained via the `yfinance` library, which provides access to Yahoo Finance data. We’ve collected Bitcoin price data spanning from **2014 to present**, including key parameters:

- **Open** 🏦: Price at the beginning of the day.
- **High** ⬆️: Highest price during the day.
- **Low** ⬇️: Lowest price during the day.
- **Close** 📉: Price at the end of the day.
- **Volume** 💰: Number of Bitcoins traded.

We used **Exploratory Data Analysis (EDA)** to visualize price trends, identify patterns, and ensure the data was clean and ready for modeling.

```python
import yfinance as yf

# Download Bitcoin price data from Yahoo Finance
data = yf.download("BTC-USD", start="2014-09-16", end="2025-12-31", interval="1d")
```

**Visualizing the Closing Price** over time helped us understand Bitcoin’s volatility and seasonal trends.



## 🔧 Feature Engineering

To improve the model's accuracy, we applied **Feature Engineering** by creating several new features from the raw data:

- **Rolling Averages** 📈: Calculated for various time horizons (e.g., 2, 5, 60, 250 days).
- **RSI (Relative Strength Index)** 🔥: A momentum indicator showing the strength of price movements.
- **MACD (Moving Average Convergence Divergence)** 📊: A trend-following indicator for identifying momentum.
- **Bollinger Bands** 💹: A volatility indicator to detect price movements.
- **Log Returns** 📉: Measures percentage change in Bitcoin’s price.

```python
# Calculating Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Adding more features like RSI, MACD, Bollinger Bands, etc.
data['RSI'] = compute_rsi(data)
```

These features help the model identify patterns in Bitcoin price movement, making predictions more accurate.


## 🤖 Machine Learning & Deep Learning Model Application

### 1. Random Forest Classifier 🌲

The **Random Forest Classifier** is an ensemble model that builds multiple decision trees and aggregates their results for better accuracy. It was trained on the features we engineered, and its performance was tuned using **GridSearchCV** to find the best parameters.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='precision', verbose=2, n_jobs=-1)
grid_search.fit(train[predictors], train["Target"])
```

The best model from **GridSearchCV** was used to predict Bitcoin’s price direction. Performance was evaluated using precision and accuracy metrics.

### 2. LSTM (Long Short-Term Memory) 🧠

The **LSTM** model is ideal for time series prediction and was built to capture Bitcoin price trends over time. It learns from past data and predicts the future price direction.

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1, activation='sigmoid'))
```

The LSTM model captures temporal dependencies, enabling it to predict future price movements based on the learned patterns from the past.



## ⚙️ Tech Stack

This project leverages the following **tools and technologies**:

### 🛠️ **Libraries & Frameworks**:
- **yfinance** 📊: For collecting historical Bitcoin data from Yahoo Finance.
- **pandas** 📂: Data manipulation and analysis.
- **numpy** 🔢: Numerical computations.
- **scikit-learn** 🧑‍💻: For machine learning algorithms like Random Forest and GridSearchCV.
- **keras** & **tensorflow** 🧠: For building and training the LSTM deep learning model.
- **matplotlib** & **seaborn** 📈: Data visualization tools.
- **GridSearchCV** 🔍: Hyperparameter tuning for optimal model performance.

### ⚙️ **Tools**:
- **PyCharm** 💻: The IDE used for code editing and version control.



## 💻 Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/bitcoin-price-prediction.git
   cd bitcoin-price-prediction
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Python scripts** to train and evaluate the models.


🔍 Feel free to explore, contribute, and give feedback. Happy coding! 🎉

