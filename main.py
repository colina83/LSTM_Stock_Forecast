import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from alpha_vantage.timeseries import TimeSeries
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()




st.title("Stock Forecast")
st.markdown("Dashboard for analyzing stock movements using different "
            "statistical methods 📈🔥")

@st.cache(persist=True)
def load_data(NYSE,NASDAQ):
    NYSE = pd.read_csv(NYSE, delimiter="\t")
    NDQ = pd.read_csv(NASDAQ, delimiter="\t")
    Stocks = pd.concat([NYSE,NDQ],ignore_index=True)
    ticker = Stocks['Symbol']
    return ticker

ticker = load_data('NYSE.txt','NASDAQ.txt')

##### Stock Symbol and date
select_ticker = st.sidebar.selectbox("Please select a Ticker", ticker)
start_date = st.sidebar.date_input('Start Date', dt.date(2015,1,1))
end_date = st.sidebar.date_input('End Date', dt.datetime.now())


@st.cache(persist=True)
def company(NYSE,NASDAQ,ticker):
    NYSE = pd.read_csv(NYSE, delimiter="\t")
    NDQ = pd.read_csv(NASDAQ, delimiter="\t")
    Stocks = pd.concat([NYSE,NDQ],ignore_index=True)
    company = Stocks.loc[Stocks['Symbol'] == ticker]
    company = company["Description"]
    company = company.to_string()

    return company.replace('0','')


st.write(company('NYSE.txt','NASDAQ.txt',select_ticker))

## Extracting Data
def day_time_series(ticker, start_date, end_date):
    """
    :param API_key: In this case I'm using my API
    :param ticker: The stock value
    :return: A time series file
    """
    API_KEY = "7B2IXHD2B1WJTQKR"

    ts = TimeSeries(key=API_KEY,output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(ticker,outputsize='full')

    time_series = data[start_date:end_date]
    time_series = pd.DataFrame(time_series['4. close'])
    time_series.rename(columns = {"4. close":'price'}, inplace=True)
    time_series.reset_index(inplace=True)
    return time_series

ts = day_time_series(select_ticker,start_date,end_date)

def time_series(time_series):
    """

    :param time_series: Time Series generated from the function date_time_series
    :return: An object (plot)
    """
    fig = px.line(time_series, x="date", y='price')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig

st.write(time_series(ts))

################### Prediction #############

@st.cache(persist=True)
def LSTM_Model(time_series):
    """

    :param time_series:
    :return:
    """
    data_o = time_series.set_index('date')
    data = data_o.values # Converting to a numpy array
    #lenght size
    training_data_len = math.ceil(len(data) * .8)
    # Scaled the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    # Create the training datasets
    train_data = scaled_data[0:training_data_len,:]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i,0]) # Past 60 values
        y_train.append(train_data[i,0]) # Value 61

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #Build the LSTM Model
    model = Sequential()
    model.add(LSTM(50,return_sequences=True, input_shape = (x_train.shape[1],1 )))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    #compile the model
    model.compile(optimizer="adam", loss = "mean_squared_error")
    #train the model
    model.fit(x_train,y_train,batch_size =1, epochs=1)

    #Create testing dataset
    test_data = scaled_data[training_data_len - 60:, :]
    #create data sets x_test and y_test
    x_test = []
    y_test = data[training_data_len:,:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i,0])

    # Convert to a Numpy Array
    x_test = np.array(x_test)
    #Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # Get Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # Get the RMSE
    rmse = np.sqrt(np.mean(predictions - y_test)**2) # Prediction
    # plot the data
    train = data_o[:training_data_len]
    valid = data_o[training_data_len:]
    valid["Predictions"] = predictions

    #Visualize

    fig = plt.figure(figsize=(16,8))
    plt.title('LSTM Stock Model')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Stock Price USD at Closing', fontsize = 18)
    plt.plot(train['price'])
    plt.plot(valid[['price','Predictions']])
    plt.legend(['Training Model', 'Validation','Prediction'], loc = 'lower right')


    # Show the predicted prices
    last_60_days = data_o[-60:].values
    #scaled the data
    las_60_days_scaled = scaler.transform(last_60_days)
    #Create an empty list
    X_test = []
    #Append the past 60 days
    X_test.append(las_60_days_scaled)
    #Convert the X_test data set to a numpy arra
    X_test = np.array(X_test)
    #Reshape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    return fig

if st.checkbox("Predict the Stock Price for Tomorrow", False):
    st.write(LSTM_Model(ts))


