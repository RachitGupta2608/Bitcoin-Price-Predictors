import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

#Downloading and saving the dataset
data = yf.download("BTC-USD", start="2014-09-16", end="2025-12-31", interval="1d")

print(data.dtypes)
data.to_csv("bitcoin_stock_data.csv")

data.index = pd.to_datetime(data.index)
data.columns = ['_'.join(col).strip() for col in data.columns.values]
print(data.columns)
data=data.rename(columns={'Date_':'Date','Close_BTC-USD':'Close','High_BTC-USD':'High','Low_BTC-USD':'Low'
                         ,'Open_BTC-USD':'Open', 'Volume_BTC-USD':'Volume'})
#Visualizing the initial data
data.plot.line(y="Close", use_index=True)
plt.show()

#Creating 2 new columns
data[('Tomorrow')] = data[('Close')].shift(-1)#creates a tomorrow column which stores the nest day's closing price

data[('Target')] = (data[('Tomorrow')] > data[('Close')]).astype(int) #creates a target column which gives a integer value90 or 1) depending on whether tomorrow's closing price is greater than today's

horizons = [2, 5, 60, 250, 1000]#number of days to consider data for
predictors = []

for horizon in horizons:
    rolling_averages = data.rolling(horizon).mean()#For each horizon,we calculate the rolling average (moving average) for all columns

    ratio_column = f"Close_Ratio_{horizon}"
    data[ratio_column] = data["Close"] / rolling_averages["Close"]#creates a new feature/column where for each day, you divide the current Close price by the rolling average Close price.

    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]#akes the sum over the past horizon days of the shifted data(shifts all the rows down by one).

    predictors += [ratio_column, trend_column]

def compute_rsi(data, window=14):#RSI (Relative Strength Index)[RSI is a momentum oscillator that measures the speed and change of price movements.]
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, fast=12, slow=26, signal=9):#MACD (Moving Average Convergence Divergence[MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
    exp12 = data['Close'].ewm(span=fast, adjust=False).mean()  # Fast EMA
    exp26 = data['Close'].ewm(span=slow, adjust=False).mean()  # Slow EMA
    macd = exp12 - exp26  # MACD Line
    signal_line = macd.ewm(span=signal, adjust=False).mean()  # Signal Line
    histogram = macd - signal_line  # MACD Histogram
    return macd, signal_line, histogram

def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


data['RSI'] = compute_rsi(data)
data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = compute_macd(data)
data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data)

data.dropna(subset=['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Bollinger_Upper', 'Bollinger_Lower'], inplace=True)

predictors += ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Bollinger_Upper', 'Bollinger_Lower']


pd.set_option("display.max_columns", None)


data = data.dropna(subset=data.columns[data.columns != "Tomorrow"])
print(data.head())

train = data.iloc[:-500]
test = data.iloc[-500:]

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [ 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier(random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           scoring='precision',
                           verbose=2,
                           n_jobs=-1)

# Fit the grid search
grid_search.fit(train[predictors], train["Target"])

# Get the best hyperparameters
print("Best hyperparameters found: ", grid_search.best_params_)

best_rf = grid_search.best_estimator_



def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.55] = 1
    preds[preds <.55] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

predictions = backtest(data, best_rf, predictors)

print(predictions["Predictions"].value_counts())

print(precision_score(predictions["Target"], predictions["Predictions"]))

print(predictions["Target"].value_counts() / predictions.shape[0])


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting GridSearchCV results (Best Hyperparameters)
results = pd.DataFrame(grid_search.cv_results_)
results_sorted = results.sort_values(by='mean_test_score', ascending=False)
# Flatten the 'params' column in the grid search results
params_df = pd.json_normalize(results_sorted['params'])

# Add the flattened parameters to the results DataFrame
results_sorted = pd.concat([results_sorted, params_df], axis=1)
print(results_sorted.columns)

# Now plot the results with the hyperparameters
plt.figure(figsize=(10, 6))
sns.barplot(x='mean_test_score', y='param_n_estimators', data=results_sorted, orient='h')
plt.xlabel('Mean Test Score')
plt.ylabel('Number of Estimators (n_estimators)')
plt.title('GridSearchCV Hyperparameters and Their Mean Test Scores')
plt.show()


# Plotting Random Forest Feature Importance
importances = best_rf.feature_importances_

# Create a DataFrame with feature names and their importances
feature_importance_df = pd.DataFrame({
    'Feature': predictors,
    'Importance': importances
})

# Sorting the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')
plt.show()









