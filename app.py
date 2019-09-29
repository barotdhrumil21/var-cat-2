from flask import Flask, render_template,request,redirect,url_for
from werkzeug import secure_filename
#import tensorflow as tf
import numpy as np
#from keras import layers
#from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
#from keras.models import Model, load_model
#from keras.preprocessing import image as immage
#from keras.utils import layer_utils
#from keras.utils.data_utils import get_file
#from keras.applications.imagenet_utils import preprocess_input
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
#from keras.initializers import glorot_uniform
#import scipy.misc
import cv2
#from sklearn.model_selection import train_test_split
import os
import sys
#import pydot
#from sklearn.externals import joblib

"""
docker run -p 8501:8501 \
  --mount type=bind,source=./model,target=./model/category-model-60.h5 \
  -e MODEL_NAME=category-model-60.h5 -t tensorflow/serving
"""
#from matplotlib.pyplot import imshow
import keras.backend as K
from keras.models import model_from_json

#K.set_image_data_format('channels_last')
#K.set_learning_phase(1)

app = Flask(__name__)

@app.route('/')
def index():
    print("rendered")
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['image']
        fname = secure_filename(f.filename)
        f.save(fname)
        print("inside")
        #K.clear_session()
        json_file = open('./model/category-model-60.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./model/category-model-60.h5")
        print("Loaded model from disk")
        img = cv2.imread(fname, 0)
        img = cv2.resize(img, (64,64))
        img = np.reshape(img, (1,64,64,1))
        x = img/255
        #np.set_printoptions(suppress=True)
        print('Input image shape:', x.shape)
        label_dict = {0: 'bottom_men_Activewear', 1: 'bottom_men_Ethnic wear', 2: 'bottom_men_Innerwear & Sleepwear',
        3: 'bottom_men_Jeans', 4: 'bottom_men_Pants', 5: 'bottom_men_Plus Size', 6: 'bottom_men_Shorts',
        7: 'bottom_men_Swimwear', 8: 'bottom_women_Activewear', 9: 'bottom_women_Bottomwear', 10: 'bottom_women_Ethnic wear',
        11: 'bottom_women_Lingerie and Sleepwear', 12: 'bottom_women_Swimwear', 13: 'men_Jackets', 14: 'men_Outerwear',
        15: 'men_Shirts', 16: 'men_Suits', 17: 'men_Tshirts', 18: 'top_men_Activewear', 19: 'top_men_Ethnic wear', 20: 'top_men_Innerwear & Sleepwear',
        21: 'top_men_Plus Size', 22: 'top_women_Activewear', 23: 'top_women_Dresses', 24: 'top_women_Ethnic wear', 25: 'top_women_Lingerie and Sleepwear',
        26: 'top_women_Swimwear', 27: 'women_Dungarees', 28: 'women_Jeans', 29: 'women_Jeggings', 30: 'women_Jumpsuits & Playsuits',
        31: 'women_Outerwear', 32: 'women_Skirts', 33: 'women_Tops'}
        preds = loaded_model.predict(x)
        confidence = np.amax(preds)
        ans = label_dict[np.argmax(preds)]
        p = "Given image has label :" + ans + " with confidence :" + confidence
        return render_template('answer.html', confidence=confidence, answer=ans)
        #return redirect(url_for('predictor',filename=fname))

@app.route('/predict/<filename>')
def predictor(filename):
    print("inside")
    #K.clear_session()
    json_file = open('./model/category-model-60.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model/category-model-60.h5")
    print("Loaded model from disk")
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (64,64))
    img = np.reshape(img, (1,64,64,1))
    x = img/255
    #np.set_printoptions(suppress=True)
    print('Input image shape:', x.shape)
    label_dict = {0: 'bottom_men_Activewear', 1: 'bottom_men_Ethnic wear', 2: 'bottom_men_Innerwear & Sleepwear',
     3: 'bottom_men_Jeans', 4: 'bottom_men_Pants', 5: 'bottom_men_Plus Size', 6: 'bottom_men_Shorts',
     7: 'bottom_men_Swimwear', 8: 'bottom_women_Activewear', 9: 'bottom_women_Bottomwear', 10: 'bottom_women_Ethnic wear',
     11: 'bottom_women_Lingerie and Sleepwear', 12: 'bottom_women_Swimwear', 13: 'men_Jackets', 14: 'men_Outerwear',
     15: 'men_Shirts', 16: 'men_Suits', 17: 'men_Tshirts', 18: 'top_men_Activewear', 19: 'top_men_Ethnic wear', 20: 'top_men_Innerwear & Sleepwear',
     21: 'top_men_Plus Size', 22: 'top_women_Activewear', 23: 'top_women_Dresses', 24: 'top_women_Ethnic wear', 25: 'top_women_Lingerie and Sleepwear',
     26: 'top_women_Swimwear', 27: 'women_Dungarees', 28: 'women_Jeans', 29: 'women_Jeggings', 30: 'women_Jumpsuits & Playsuits',
     31: 'women_Outerwear', 32: 'women_Skirts', 33: 'women_Tops'}

    preds = loaded_model.predict(x)
    confidence = np.amax(preds)
    ans = label_dict[np.argmax(preds)]
    p = "Given image has label :" + ans + " with confidence :" + confidence
    return p

if __name__ == '__main__':
    print("started")
    port = int(os.environ.get('PORT'))
    app.run(host='0.0.0.0', port=port)
