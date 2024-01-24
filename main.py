import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import cv2

import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow_addons as tfa

import keras_tuner
import os
import importlib
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from scripts.variables import *
from scripts.modules import *