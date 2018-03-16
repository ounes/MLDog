#! /usr/bin/env python

import json
import sys


# Dependencies can be installed by running:
# pip install keras tensorflow h5py pillow


# Run script as:
# ./extract-features.py images/*.jpg

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import numpy as np

def get_model():
    # Load model VGG16 as described in https://arxiv.org/abs/1409.1556
    # This is going to take some time...
    base_model = VGG16(weights='imagenet')
    # Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    # (see the paper above for mode details)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    return model

def main():

    # Load model VGG16 as described in https://arxiv.org/abs/1409.1556
    # This is going to take some time...

    base_model = VGG16(weights='imagenet')

    # Model will produce the output of the 'fc2'layer which is the penultimate neural network layer
    # (see the paper above for mode details)

    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)


    # For each image, extract the representation
    for image_path in sys.argv[1:]:
        features = extract_features(model, image_path)

        with open(image_path + ".json", "w") as out:
            json.dump(features, out)


def extract_features_from_binary(model, image_data):
    from PIL import Image
    from io import BytesIO
    byteIO_image_data = BytesIO(image_data)
    pil_image = Image.open(byteIO_image_data)
    pil_image = pil_image.resize((224,224))
    x = image.img_to_array(pil_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features.tolist()[0]

def extract_features_from_binary2(image_data):
    model = get_model()
    from PIL import Image
    from io import BytesIO
    byteIO_image_data = BytesIO(image_data)
    pil_image = Image.open(byteIO_image_data)
    pil_image = pil_image.resize((224,224))
    x = image.img_to_array(pil_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features.tolist()[0]

def extract_features(model, image_path):

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features.tolist()[0]


if __name__ == "__main__":
    main()
