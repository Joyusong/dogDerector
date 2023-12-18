from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from predict_model import video_predict
sess = tf.compat 
app = Flask(__name__)
os.environ['TF_VERSION'] = '2.15.0'
def predict_video(video):
    pre = video_predict(video)
    return pre

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)  # 서버에 파일 저장
        re=  predict_video(uploaded_file.filename)
    return render_template('result.html',result = re)

if __name__ == '__main__':
    app.run(debug=True)
