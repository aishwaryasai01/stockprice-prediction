from Cython import inline
from flask import Flask ,request,redirect, render_template
import flask
import plotly.graph_objects as go
from keras import Sequential
from plotly.subplots import make_subplots
import pandas as pd

import time
from datetime import datetime
import datetime
import pandas as pd

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from keras.models import sequential
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array
app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Hello a"

# @app.route('/')
# def index():
#     """ Displays the index page accessible at '/'
#     """
#     return render_template('index.html')
@app.route('/')
def main():
    return redirect('/userform')

@app.route('/userform',methods=['GET'])
def userstockform():
    return render_template('ytindex.html')
#
# @app.route('/<name>')
# def get_product(name):
#   return "Hello " + str(name)

# @app.route('/stock')
# def get_stock():
#   return "Hello This page is for Stock Data"


#web scraping
# @app.route('/stockchart/<stockname>')
@app.route('/stockchart', methods =["POST"])
def get_stock_chart():
    ticker = request.form.get("cname")
    if request.form.get("sdate") != '' and request.form.get("edate") != '':
        temp_end_date = (request.form.get("edate"))
        temp_start_date = (request.form.get("sdate"))
        end_date = datetime.datetime.strptime(temp_end_date, '%Y-%m-%d')
        start_date = datetime.datetime.strptime(temp_start_date, '%Y-%m-%d')
    #elif request.form.get("sdate") == '' and request.form.get("edate") != '':
    #    temp_end_date = (request.form.get("edate"))
    #    end_date = datetime.datetime.strptime(temp_end_date, '%Y-%m-%d')
    #    start_date = end_date - datetime.timedelta(days=365 * 5)
    else:
        end_date = datetime.datetime.today()
        start_date = end_date - datetime.timedelta(days=365*5)  # 1 year

    # To Convert Date to Unix Date Format
    period1 = int(time.mktime(start_date.timetuple()))
    period2 = int(time.mktime(end_date.timetuple()))
    #print(period1)
    #print(period2)
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    d = pd.read_csv(query_string)

    def prediction(df,nod):
        print("NullValuePresent: ", df.isnull().sum())

        # target variable
        df1 = df.reset_index()['Close']
        plt.plot(df1)
        plt.xlabel('Days')
        plt.ylabel('Closing price')
        plt.show()

# selecting features

# Scaling Features to lie b/w 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# splitting test and train set
        training_size = int(len(df1) * 0.65)
        test_size = len(df1) - training_size
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]



        time_step=100
        dataX, dataY = [], []
        for i in range(len(train_data) - time_step - 1):
            a = train_data[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(train_data[i + time_step, 0])
        dataX,dataY=np.array(dataX), np.array(dataY)




# reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = dataX.reshape(dataX.shape[0], dataX.shape[1], 1)
        y_train=dataY

        dataX,dataY=[],[]
        for i in range(len(test_data) - time_step - 1):
            a = test_data[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(train_data[i + time_step, 0])
        x_test,y_test=np.array(dataX), np.array(dataY)
        X_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1)
### Lets Do the prediction and check performance metrics

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
##Transformback to original form

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
        print(math.sqrt(mean_squared_error(y_train, train_predict)))
### Test Data RMSE
        print(math.sqrt(mean_squared_error(y_test, test_predict)))
### Plotting
# shift train predictions for plotting
        look_back = 100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
# shift test predictions for plotting
        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
# plot baseline and predictions


    #future predictions
        x_input = test_data[341:].reshape(1, -1)
        print(x_input.shape)
        #print(len(test_data))
        #
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        # demonstrate prediction for next 10 days
        lst_output = []
        n_steps = 100
        i = 0
        while (i < int(nod)):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)

                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.extend(yhat.tolist())

            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)

                temp_input.extend(yhat[0].tolist())

                lst_output.extend(yhat.tolist())
            i = i + 1
            #
        #print(len(lst_output))
        day_new = np.arange(1, 101)
        day_pred = np.arange(101, 101+int(nod))
        #
        print(len(df1))
        #
        plt.plot(scaler.inverse_transform(df1),label='actual data')
        plt.plot(trainPredictPlot, label='Train prediction')
        plt.plot(testPredictPlot, label='Test prediction')
        plt.xlabel('Days')
        plt.ylabel('Closing price')
        plt.legend(loc="lower right")
        plt.show()


        #
        df3=df1.tolist()
        df3.extend(lst_output)
        df3=scaler.inverse_transform(df3).tolist()
        plt.plot(df3[-len(lst_output):],label='prediction')
        plt.show()
        # split into samples
        print((df3[-len(lst_output)]).shape[1])
        print(df3[-len(lst_output):])
    if request.form.get("nodp")!='':
        no_of_days=request.form.get("nodp")
        prediction(d,no_of_days)
    # --- Creating Chart
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # include candlestick with rangeselector
    fig.add_trace(go.Candlestick(x=d['Date'],
                                 open=d['Open'], high=d['High'],
                                 low=d['Low'], close=d['Close']),
                  secondary_y=True)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()

    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # return render_template('notdash.html', graphJSON=graphJSON, header=header,description=description)
    return render_template('notdash.html')


@app.route('/candle')
def candle_chart():
    df = pd.read_csv('stock.csv')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # include candlestick with rangeselector
    fig.add_trace(go.Candlestick(x=d['Date'],
                                 open=d['Open'], high=d['High'],
                                 low=d['Low'], close=d['Close']),
                  secondary_y=True)

    # include a go.Bar trace for volumes
    # fig.add_trace(go.Bar(x=df['Date'], y=df['Volume']),
    #                secondary_y=False)

    # fig.layout.yaxis2.showgrid=True

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()
    # return (df)
    return render_template('notdash.html', graphJSON=graphJSON, header=header,description=description)


if __name__ == "__main__":
    app.run( port=8000, debug=True)

    # if __name__ == '__main__':
    #     app.run(host='0.0.0.0', port=80)