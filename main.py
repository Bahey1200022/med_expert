import cv2
from flask import Flask , request, render_template,jsonify
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import base64
import torch
import config
import models
from io import BytesIO
from PIL import Image
app = Flask(__name__)

def decodefromjs(data_url):
    # Assuming this function decodes the base64 string to a numpy array
    encoded_data = data_url.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return nparr

    
    
    
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    data_url1 = request.json['image_data']
    global img
    pic = []

    img = cv2.imdecode(decodefromjs(data_url1), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #apply prediction
    model = models.load_model('Trial_TL2.pth', config.device)
    
    image_tensor, original_image = models.preprocess_image(img)
    
    output_mask = models.segment_image(model, image_tensor, config.device)
    output_mask = (output_mask * 255).astype(np.uint8)

    
    
    
    
    
    
    

    # Encode image to base64
    _, buffer = cv2.imencode('.png', output_mask)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    

    return jsonify({ 'image_data': img_base64})
    
    
    
    
    
    



def decodefromjs(data_url):
        image_data = data_url.split(',')[1]
        

    # Decode the image data from base64
        decoded_data = base64.b64decode(image_data)
    
    # Convert the decoded data to a NumPy array
        np_data = np.frombuffer(decoded_data, np.uint8)
        return np_data

    
if __name__ == '__main__':
    app.run(debug=True,port=5000)