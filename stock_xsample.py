import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model
import matplotlib.dates as mdates
import pandas_datareader.data as web
import datetime as dt

def predict_stock_trend(ticker, start, end, train_sample, train_epoch, forecast_days):
    #start = dt.datetime(2015, 1, 1)
    #end = dt.datetime(2021, 1, 25)
    #ticker = 'TSLA'

    df = web.DataReader(ticker, 'yahoo', start, end)

    data = df['Adj Close'].values
    #print(data)

    #scale input between 0, 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    #print(scaled_data)
    #print(scaled_data.shape[0])

    x = []
    y = []
    sample = train_sample

    #y lags x by n samples
    for i in range(sample, scaled_data.shape[0]):
        x.append(scaled_data[i-sample:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    #print(x.shape)
    #(1208, n, 1)
    #print(x[0])
    print(y.shape)
    #print(y)

    #train on GPU
    pysical_devices = tf.config.experimental.list_physical_devices('GPU')
    #print("Num GPUs Available: ", len(pysical_devices))
    tf.config.experimental.set_memory_growth(pysical_devices[0], True)

    model = Sequential()
    model.add(LSTM(128, input_shape=(sample, 1), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(128))
    model.add(Dropout(0.1))

    model.add(Dense(1))
    model.summary()

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
    )

    model.fit(x,
              y,
              epochs=train_epoch,
              batch_size=32)

    model.save("stock_xsample.h5")

    x_forecast = scaled_data.flatten()
    #print(x_forecast.shape)
    #print(x_forecast[-50:])

    x_forecast = x_forecast[-sample:]
    y_forecast = np.array([])
    forecast_period = forecast_days
    #print(x_forecast)

    #model = load_model('stock_xsample.h5')

    #assume sample = 50. use last 50 known stock price to predict the next one
    #then use last 49 known stock price + previous predicted price to predict next one...
    #loop for forecast peroid, all forecasted price are recorded in y_forecast
    for i in range(0, forecast_period):
        if i < sample:
            x_forecast_new = np.append(x_forecast[i:], y_forecast)
        else:
            x_forecast_new = y_forecast[i-sample:]

        x_forecast_reshaped = np.reshape(x_forecast_new, (1, sample, 1))
        predicted_stock_price_single = model.predict(x_forecast_reshaped)
        y_forecast = np.append(y_forecast, predicted_stock_price_single[0][0])

    #print(y_forecast)
    #print(predicted_stock_price.shape)
    #print(predicted_stock_price)

    y_forecast = np.reshape(y_forecast, (len(y_forecast), 1))
    predicted_stock_price = scaler.inverse_transform(y_forecast)
    predicted_stock_price = predicted_stock_price.flatten()
    #print(predicted_stock_price)

    ax = plt.figure(figsize=(7, 5), dpi=100).add_subplot(111)
    #fig, ax = plt.subplots()

    t = df['Adj Close'].index
    t = pd.to_datetime(t)
    #print(t)

    #create xaxis for predicted stock price
    t_predicted = pd.bdate_range(end.strftime('%Y-%m-%d'), periods=forecast_period, freq="D")
    #print(t_predicted)

    ax.plot(t_predicted, predicted_stock_price, label="Predicted stock price", color="red")
    ax.plot(t, data, label="Real Stock Price", color="blue")

    ax.legend()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=300))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Y'))

    ax.set_title(ticker + " Price Prediction")
    plt.ylabel("USD")

    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")

    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=.2)

    zoom_factory(ax)
    plt.show()


# for mouse scroll zoom
def zoom_factory(ax,base_scale = 1.2):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        # set the range
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print (event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        ax.figure.canvas.draw_idle() # force re-draw the next time the GUI refreshes

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun


#predict_stock_trend(ticker, start, end, train_sample, train_epoch, forecast_days)
predict_stock_trend("FDX", dt.datetime(2015, 1, 1), dt.datetime.today(), 50, 100, 90)

