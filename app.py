import streamlit as st 
st.title("Image Segmentation")

st.write("Packages Loading....")

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.pyplot import imread
style.use('fivethirtyeight')
from PIL import Image
import numpy as np
import os
import time
#from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

st.write("Packages Loaded Successfully!")

json_file=open("model.json","r")
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

img_size = (160, 160)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image,caption="Uploaded image",use_column_width=True)
	image = image.resize(img_size)
	#pil_image = Image.open(uploaded_file).convert('RGB')
	image_val = np.array(image)
	x=np.zeros((1,)+img_size+(3,),dtype="uint8")
	x[0]=image_val
	st.write("Predicting....")
	my_bar = st.progress(0)
	for percent_complete in range(80):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
	val_preds = loaded_model.predict(x)
	mask = np.argmax(val_preds[0], axis=-1)
	mask = np.expand_dims(mask, axis=-1)
	img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
	for percent_complete in range(80,100):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
	st.image(img, caption='Predicted image', use_column_width=True)
	st.write("")