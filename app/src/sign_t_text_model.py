import os
import numpy as np
import joblib
import pickle
import cv2
import pandas as pd

from keras.models import load_model
from typing_extensions import final
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier


current_path =os.path.dirname(os.path.abspath(__file__))



def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        encoder = pickle.load(file)
    return encoder


label_en=load_pickle("artifact/model/randomforest/label_encoder.pkl")


label_bz=load_pickle("artifact/model/randomforest/label_encoder.pkl")




rf_model = joblib.load("artifact/model/randomforest/random_forest_model_scaled_XYZ.joblib")

def random_forest(input_list):
  input_list=(np.array(input_list).reshape(1,-1))/300
  predict=rf_model.predict(input_list)
  return label_en[predict[0]]




# Load the model from the file
gb_model = joblib.load("artifact/model/gradian_boost/gbm.joblib")

def gradien_boost(input_list):
  input_list=(np.array(input_list).reshape(1,-1))/300
  predict=gb_model.predict(input_list)
  return label_en[predict[0]]



# Load the model from the file
cat_model = joblib.load("artifact/model/cat_boost/cat.joblib")

def cat_boost(input_list):
  input_list=(np.array(input_list).reshape(1,-1))/300
  predict=cat_model.predict(input_list)
  return label_en[predict[0][0]]


# Load the model from the file
xgb_model = joblib.load("artifact/model/cat_boost/cat.joblib")

def xgb_boost(input_list):
  input_list=(np.array(input_list).reshape(1,-1))/300
  predict=xgb_model.predict(input_list)
  return label_bz[predict[0][0]]





'''

# Load the model
sign_CNN_model = load_model(os.path.join(current_path,"model/CNN_sign_language/CNN_sign_language.hdf5"))

classes_3=['A', 'B', 'C' ,'D' ,'E' ,'F', 'G' ,'H' ,'I', 'J' ,'K' ,'L' ,'M', 'N' ,'O' ,'P', 'Q' ,'R',
 'S' ,'T' ,'U' ,'V' ,'W' ,'X' ,'Y' ,'Z' ,'del' ,'space']

def CNN_model(image,tresh=0.01):
  input_shape = (224, 224, 3)

  #image=cv2.imread(image_path)
  # Resize, scale and reshape image before making predictions
  resized = cv2.resize(image, (224,224))
  resized = (resized / 255.0).reshape(-1,input_shape[1],input_shape[0],input_shape[2])
  #cv2_imshow((resized[0] * 255).astype(np.uint8))

  predict=sign_CNN_model.predict(resized)
  #print(predict)

  pred=zip(classes_3,predict[0])
  pre_dict = dict(pred)
  final_dict={}

  for i in range(len(pre_dict)):
    if list(pre_dict.values())[i] >= tresh:
      final_dict[list(pre_dict.keys())[i]]=list(pre_dict.values())[i]
    else:
      pass

  #print(final_dict)
  if final_dict is not None:
    # Sort the dictionary by values
    sorted_dict = dict(sorted(final_dict.items(), key=lambda item: item[1],reverse=True))
    #print(list(sorted_dict.items()))
    return list(sorted_dict.items())
  else:
    return None
  
'''
