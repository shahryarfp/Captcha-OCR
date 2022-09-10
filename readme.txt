Author: Shahryar Namdari
@AIMedic

#Required dataset **IMPORTANT**
> The dataset folder(samples) which contains 1040 .png captcha images must be in a same directory as the .py code
> It is important that the name of dataset folder have to be: 'samples'
> otherwise, the variable named "path_to_the_samples_folder" inside the code, should be changed to the address of dataset folder
> In google colab if you want to use your data on google drive, you can uncomment the part of code which is for connecting to google drive

# Imported libraries in the code:
import os
from math import sqrt,floor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Explanation of the Code
> This code creates and trains a Deep Learning model by the use of captcha dataset. This code ran successfully on Google Colab.
> Best accuracy of the test data is 100% for 104 test captchas
> All explanations of the code is available in the notebook file
> Also .html and .pdf format of notebook file is available
> You have to just run the code and get evaluation files in the current directory of the code
> evaluation files are:
	model_accuracy.txt 	==> Calibrated in percentage
	model_summary.txt  	==> Illustrates the structure of the model
	model_loss.png		==> Shows the training and validation loss in training process
	few_training_data_with_label.png
	few_correctly_predicted_data_with_label.png ==> (if needed)
	few_wrongly_predicted_data_with_label.png ==> (if needed)
sample output files are available in the "sample output" folder. this files are for a model with 98% accuracy.


# Command & Output
command:
you have to just simply run the .py file without any input
output:
If you are using GPU, evaluation files will be created in less than 4 minutes in the current directory of the code.
Using CPU will take few more minutes


# Reference
One of the main sources that helped me with this project:
https://keras.io/examples/vision/captcha_ocr/
I also used CTC Layer class defined in the link above