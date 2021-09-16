import tensorflow as tf
import argparse
import requests
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import json

def imagepath_to_tfserving_payload(img_path):    
    img = image.img_to_array(image.load_img(img_path, target_size=(224, 224)))
    X = np.expand_dims(img, axis=0).astype('float32')
    X = preprocess_input(X)
    payload = dict(instances=X.tolist())
    payload = json.dumps(payload)
    return payload

def tfserving_predict(image_payload, url):
#     url = 'http://localhost:8501/v1/models/resnet:predict'
    r = requests.post(url, data=image_payload)
    pred_json = json.loads(r.content.decode('utf-8'))
    return pred_json['predictions']

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='Tensorflow server url', default='http://localhost:8501/v1/models/resnet:predict', type=str)
    parser.add_argument('--image_path', help='input image', type=str)
    args = parser.parse_args()
    
    preprocess_image=imagepath_to_tfserving_payload(args.image_path)
    Embeddings=tfserving_predict(preprocess_image,args.url)
    print('Embeddings:',Embeddings[0])
    
if __name__ == '__main__':
    run()