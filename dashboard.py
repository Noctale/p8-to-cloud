#import pandas as pd
import streamlit as st
import requests
from PIL import Image
#import io
import json
import numpy as np
#from method import array_to_img
#import tensorflow as tf
#import matplotlib.pyplot as plt

def array_to_img(x, data_format=None, scale=True, dtype=None):

  if data_format is None:
    data_format = "channels_last"#backend.image_data_format()
  if dtype is None:
    dtype = "float32"#backend.floatx()
  #if pil_image is None:
    #raise ImportError('Could not import PIL.Image. '
    #                  'The use of `array_to_img` requires PIL.')
  x = np.asarray(x, dtype=dtype)
  if x.ndim != 3:
    raise ValueError('Expected image array to have rank 3 (single image). '
                     f'Got array with shape: {x.shape}')

  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError(f'Invalid data_format: {data_format}')

  # Original Numpy array x has format (height, width, channel)
  # or (channel, height, width)
  # but target PIL image has format (width, height, channel)
  if data_format == 'channels_first':
    x = x.transpose(1, 2, 0)
  if scale:
    x = x - np.min(x)
    x_max = np.max(x)
    if x_max != 0:
      x /= x_max
    x *= 255
  if x.shape[2] == 4:
    # RGBA
    return pImage.fromarray(x.astype('uint8'), 'RGBA')
  elif x.shape[2] == 3:
    # RGB
    return Image.fromarray(x.astype('uint8'), 'RGB')
  elif x.shape[2] == 1:
    # grayscale
    if np.max(x) > 255:
      # 32-bit signed integer grayscale image. PIL mode "I"
      return Image.fromarray(x[:, :, 0].astype('int32'), 'I')
    return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
  else:
    raise ValueError(f'Unsupported channel number: {x.shape[2]}')   
    
    
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
    img = array_to_img(printable_pred)
    

    return img
    
#def just_show(img_path):

    #data_in = {'file': open(img_path ,'rb')}
    
    #response = requests.post('http://127.0.0.1:8000/predict', files=data_in)

    #if response.status_code != 200:
        #raise Exception(
            #"Request failed with status {}, {}".format(response.status_code, response.text))
            
    #image_bytes = io.BytesIO(response.content)
    #img = Image.open(image_bytes)

    #return img


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
