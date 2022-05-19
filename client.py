import pickle
import numpy as np
import json
import tensorflow as tf
import requests
import PIL
import matplotlib.pyplot as plt

url = 'http://127.0.0.1:5000/model'
path = 'D:\Deakin\SIT744_DeepLearning\\assign2\\archive (2)\garbage_classification\\battery\\battery1.jpg'
img = np.array(tf.keras.preprocessing.image.load_img(path).resize((150,150))).tolist()
data = json.dumps({'img':img})
response = requests.post(url, data)
print('the prediction is ', response.text)

