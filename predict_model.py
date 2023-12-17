#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np
import cv2

import tensorflow as tf
import os

result = []
graph_def = tf.compat.v1.GraphDef()
labels = []
# These are set to the default names from exported models, update as needed.
filename = "model.pb"
labels_filename = "labels.txt"
# Import the TF graph
with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
 # Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())
exif_orientation_tag = 0x0112

def predict(image):
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    image = opencv_image
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    image = cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)
    return image



def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]



def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)



def video_predict(video_name):
    cap = cv2.VideoCapture(video_name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    images = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        images.append(im)
    for i, image in enumerate(images):
        image = predict(image)
        if image is not None:
            h, w = image.shape[:2]
            min_dim = min(w,h)
            max_square_image = crop_center(image, min_dim, min_dim)


            augmented_image = resize_to_256_square(max_square_image)


            with tf.compat.v1.Session() as sess:
                input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
            network_input_size = input_tensor_shape[1]

            augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

            output_layer = 'loss:0'
            input_node = 'Placeholder:0'

            with tf.compat.v1.Session() as sess:
                try:
                    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
                    predictions = sess.run(prob_tensor, {input_node: [augmented_image] })
                except KeyError:
                    print ("Couldn't find classification output layer: " + output_layer + ".")
                    print ("Verify this a model exported from an Object Detection project.")
                    exit(-1)


            n = 29
            index = np.array(predictions).argsort()[-n:]
            """
            for i in range(5):
                print(i+1, "tag: " , labels[index[0][-(i+1)]])
                print()
            """
            result.append(labels[index[0][-1]])
    return result

