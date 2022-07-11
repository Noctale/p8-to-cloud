import pandas as pd
import streamlit as st
import requests
from PIL import Image
import io
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

   
    
    
def request_prediction(img_path):

    data_in = {'file': open(img_path ,'rb')}
    
    response = requests.post('http://127.0.0.1:8000/mask_predict', files=data_in)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
            
    decodedArrays = json.loads(response.content)
    y_pred = np.asarray(decodedArrays["array"])
    printable_pred = np.argmax(y_pred, axis=2)
    printable_pred = np.expand_dims(printable_pred, axis=-1)
    img = tf.keras.preprocessing.image.array_to_img(printable_pred)
    

    return img
    
def just_show(img_path):

    data_in = {'file': open(img_path ,'rb')}
    
    response = requests.post('http://127.0.0.1:8000/predict', files=data_in)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
            
    image_bytes = io.BytesIO(response.content)
    img = Image.open(image_bytes)

    return img


def main():
    img_api = 'http://127.0.0.1:8000/predict'


    st.title('Mask Prediction')

    img_path = st.text_input('Liens vers votre image')



    predict_btn = st.button('Prédire')
    if predict_btn:
        pred = None
        pred = request_prediction(img_path)

        st.image(pred, channels = 'BGR')

#st.download_button(
   #  label="Download data as CSV",
   #  data=csv,
   #  file_name='large_df.csv',
   #  mime='text/csv',
 #)

if __name__ == '__main__':
    main()
