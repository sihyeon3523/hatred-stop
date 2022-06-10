from django.conf import settings
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # pillow
import cv2 # opencv-python


def predict_emotion(path):

    base_url = settings.MEDIA_ROOT_URL + settings.MEDIA_URL
    model_url = base_url + ''
    model = models.load_model(model_url, compile=False)

    sample_one =
    predict_result =

    words_dict = {

    }

    return words_dict.get(predict_result)
