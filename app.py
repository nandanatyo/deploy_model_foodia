from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# #from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50

# LIBRARY
import pandas as pd
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from paddleocr import PaddleOCR,draw_ocr

# IMAGE PROCESSING
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
model = load_model('bilstm_model2.h5')


# -------------------------------------------------------------------

from img_to_txt_ocr import prepro
from model import model

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "images/" + imagefile.filename
    imagefile.save(image_path)

    df_test_tokens, X_test = prepro(image_path)
    hasil_bilstm_model = model(df_test_tokens, X_test)

    hasil_bilstm_model.to_csv('predicted_results/hasil_'+ imagefile.filename + '.csv', index=False)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=3000, debug=True)