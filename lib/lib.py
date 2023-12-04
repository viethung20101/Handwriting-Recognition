# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Commonly used modules
import numpy as np
import os
import sys

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="keras")
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)