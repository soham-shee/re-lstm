# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import streamlit as st
import pickle
import base64
from io import BytesIO

# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real Demand')
    plt.plot(predicted, color='green',label='Predicted Demand')
    plt.title('Demand Prediction')
    plt.xlabel('Time')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()

# Calculation of Root Mean Squared-Error

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))



# def createGRUModel(df):
#     prev=st.text_input('Enter the past lookup days : ')
#     num=int(prev)
#     training_set = list(df.values)
#     training_set=training_set.reshape((training_set.shape[0],1))
#     sc = MinMaxScaler(feature_range=(0,1))
#     training_set_scaled = sc.fit_transform(training_set)
#     X_train = []
#     Y_train = []
#     for i in range(num,training_set.shape):
#         X_train.append(training_set_scaled[i-num:i,0])
#         Y_train.append(training_set_scaled[i,0])
#     X_train, Y_train = np.array(X_train), np.array(Y_train)

# st.markdown("""
#     <style>
#     body {
#         .center: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)
st.title("Load Forecasting Using LSTM (Testing)")

uploaded_files = st.file_uploader("Upload file", type=["xls", "xlsx"], accept_multiple_files=True)
file = st.file_uploader('Model file .h5 model', type='.h5')
if uploaded_files and file:
    model_bytes = file.read()
    model = pickle.load(BytesIO(model_bytes))
    dfs = []
    
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        st.write(f"Filename: {uploaded_file.name}")
        df = df[['Date N Time', 'M1_PWR']]
        # st.dataframe(df)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['Date N Time'] = pd.to_datetime(combined_df['Date N Time'])
    combined_df.set_index('Date N Time', inplace=True)
    combined_df = combined_df.sort_index()
    
    # st.write("Combined and Sorted DataFrame:")
    # st.dataframe(combined_df)

    combined_df = combined_df['M1_PWR'].resample('6min').mean().to_frame()
    st.line_chart(combined_df['M1_PWR'])

    # Testing Part
    test_set=combined_df.values
    test_set=test_set.reshape((test_set.shape[0],1))
    st.write(test_set.shape)

    sc = MinMaxScaler(feature_range=(-1, 1))
    test_set_scaled=sc.fit_transform(test_set)
    inputs = test_set_scaled

    # X_test = []
    # for i in range(100):
        
    inputs = test_set_scaled[-100:]  # shape: (100, 1)

    # Convert to list to allow appending predicted values
    temp_input = list(inputs.flatten())  # flatten to (100,) list
    predictions = []

    # Perform 100 recursive predictions
    for i in range(100):
        # Take last 100 values (may include predictions from earlier steps)
        input_seq = temp_input[-100:]

        # Reshape to (1, 100, 1) for model prediction
        input_array = np.array(input_seq).reshape(1, 100, 1)

        # Predict next value
        predicted_data = model.predict(input_array, verbose=0)
        predicted_data = sc.inverse_transform(predicted_data)  # shape (1, 1)

        # Extract scalar value
        predicted_value = predicted_data[0][0]

        # Store prediction
        predictions.append(predicted_value)

        # Append prediction to temp_input for next iteration
        temp_input.append(predicted_value)

    # Optional: show the predictions
    predictions=np.array(predictions)
    st.write(predictions)


    # for i in range(len(inputs)):
    #     X_test.append(inputs[i, 0])
    # X_test=np.array(X_test)
    # X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    # st.write(X_test.shape)

    # for i in range(100,len(inputs)):
    #     X_test.append(inputs[i-100:i,0])
    # X_test = np.array(X_test)
    # X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    # predicted_data = model.predict(X_test)
    # predicted_data = sc.inverse_transform(predicted_data)
    # print("test shape - ",X_test.shape)
    # st.write("Predicted Values : ")
    # st.write(predicted_data)



# df = st.file_uploader("Upload file", type={"csv"})
# file = st.file_uploader('Model (.h5) file', type='.h5')
# if df and file is not None:
#     df = pd.read_csv(df, index_col=[0], parse_dates=[0])
#     # model=pickle.load(open(file,'rb'))
#     model_bytes = file.read()
#     model = pickle.load(BytesIO(model_bytes))
#     st.write("Shape : ",df.shape)
#     st.write("First 10 rows : ")
#     st.write(df.head(10))
#     df=df['nat_demand'].resample('D').mean()
#     values = st.slider(
#     "Select a range for testing values",
#     0, df.shape[0],(0,df.shape[0]-10))
#     # st.write("Values:", values)
#     start=values[0]
#     end=values[1]
#     # st.write(df.shape[0])
#     # Ploting the graph
#     fig=plt.figure(figsize=(16,6))
#     plt.title('Graph of Net Demand vs Daily data')
#     plt.plot(df[start:end])
#     st.pyplot(fig)


#     # Main functions
#     df_original=df
#     df=df[start:end]
#     prev = st.slider('Past lookup days (less than ending and starting dates) : ', key=2, min_value=1, max_value=200, value=1, step=1)
#     test_set = np.array(df)
#     test_set=np.reshape(test_set, (test_set.shape[0],1))
#     num=test_set.shape[1]
    
#     # After :
#     # sc = MinMaxScaler(feature_range=(0,1))
#     # dataset_total = pd.concat((df[:end],df[start:]),axis=0)
#     # inputs = np.array(dataset_total[len(dataset_total)-len(test_set) - prev:])
#     # inputs = inputs.reshape((-1,1))
#     # inputs  = sc.fit_transform(inputs)

#     sc = MinMaxScaler(feature_range=(0,1))
#     training_set_scaled = sc.fit_transform(test_set)
#     inputs = df[len(test_set) - prev:].values
#     inputs = inputs.reshape(-1,1)
#     inputs  = sc.transform(inputs)

#     X_test = []
#     for i in range(prev,prev+num):
#         X_test.append(inputs[i-prev:i,0])
#     # st.write(X_test[0])
#     X_test = np.array(X_test)
#     X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
#     predicted_data = model.predict(X_test)
#     predicted_data = sc.inverse_transform(predicted_data)
#     prediction=int(predicted_data[0])
#     col1, col2 = st.columns(2, gap="medium")
#     with col1:
#         st.write("Predicted Next Value : ")
#         st.write(prediction)
#     with col2:
#         st.write("Original Next Value : ")
#         st.write(df_original.iloc[end+1])

#     # st.write(plot_predictions(test_set,predicted_data))
#     # st.write(return_rmse(test_set,predicted_data))
