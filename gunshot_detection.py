#!/usr/bin/env python
# coding: utf-8

# Package Imports #
 
import os
import pyaudio
import sys
import librosa
import time
import scipy.signal
import numpy as np
import tensorflow as tf
import six
import tensorflow.keras as keras
from threading import Thread
from datetime import timedelta as td
from queue import Queue
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import backend as K
from tkinter import *
from time import sleep
import io
from contextlib import redirect_stdout
from helpers import *
from constants import *



window = Tk()
window.configure(bg="red")
var = StringVar()
soundclass = Label(window, textvariable=var)
soundclass.config(font=("Courier", 90))

soundclass.pack(padx=200, pady=400)


# Stream Variables
sound_data = np.zeros(0, dtype = "float32")
noise_sample_captured = False
gunshot_sound_counter = 1
noise_sample = []
audio_analysis_queue = Queue()

# HARDCODED PATH
DIRPATH = os.getcwd()


# Loading in Augmented Labels #
labels = np.load(os.path.join(DIRPATH, "augmented_labels.npy"))


# Binarizing Labels #

labels = np.array([("gun_shot" if label == "gun_shot" else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))


# Loading the Models #
    
# Loads 44100 x 1 Keras model from H5 file
interpreter_1 = tf.lite.Interpreter(model_path = os.path.join(DIRPATH, "models/1D.tflite"))
interpreter_1.allocate_tensors()
    
# Sets the input shape for the 44100 x 1 model
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_shape_1 = input_details_1[0]['shape']

# Loads 128 x 64 Keras model from H5 file
interpreter_2 = tf.lite.Interpreter(model_path = os.path.join(DIRPATH, "models/128_x_64_2D.tflite"))
interpreter_2.allocate_tensors()

# Gets the input shape from the 128 x 64 Keras model
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
input_shape_2 = input_details_2[0]['shape']

# Loads 128 x 128 Keras model from H5 file
interpreter_3 = tf.lite.Interpreter(model_path = os.path.join(DIRPATH, "models/128_x_128_2D.tflite"))
interpreter_3.allocate_tensors()

# Gets the input shape from the 128 x 128 Keras model
input_details_3 = interpreter_3.get_input_details()
output_details_3 = interpreter_3.get_output_details()
input_shape_3 = input_details_3[0]['shape']


previous_data = None

## Callback Thread ##
def callback(in_data, frame_count, time_info, status):
    global sound_data
    sound_buffer = np.frombuffer(in_data, dtype="float32")
    sound_data = np.append(sound_data, sound_buffer)
    if len(sound_data) >= 88200:
        sound_data = sound_data[-88200:]
        audio_analysis_queue.put(sound_data)
        current_time = time.ctime(time.time())
        audio_analysis_queue.put(current_time)
        
    return sound_buffer, pyaudio.paContinue


pa = pyaudio.PyAudio()

stream  = pa.open(format = AUDIO_FORMAT,
                    rate = AUDIO_RATE,
                    channels = NUMBER_OF_AUDIO_CHANNELS, \
                    input_device_index = AUDIO_DEVICE_INDEX,
                    input = True, \
                    frames_per_buffer=NUMBER_OF_FRAMES_PER_BUFFER, 
                    stream_callback=callback)

# Starts the callback thread
stream.start_stream()



# This thread will run indefinitely
while True:

    # Gets a sample and its timestamp from the audio analysis queue
    microphone_data = np.array(audio_analysis_queue.get(), dtype = "float32")
    time_of_sample_occurrence = audio_analysis_queue.get()

    # Finds the current sample's maximum frequency value
    maximum_frequency_value = np.max(microphone_data)
        
    # Determines whether a given sample potentially contains a gunshot
    if maximum_frequency_value >= AUDIO_VOLUME_THRESHOLD:
        
        

        start_time = time.time()
        
        # Post-processes the microphone data
        modified_microphone_data = librosa.resample(y = microphone_data, orig_sr = AUDIO_RATE, target_sr = 22050)
        print("--- %s seconds ---" % (time.time() - start_time))
        if NOISE_REDUCTION_ENABLED and noise_sample_captured:
                # Acts as a substitute for normalization
                modified_microphone_data = remove_noise(audio_clip = modified_microphone_data, noise_clip = noise_sample)
                number_of_missing_hertz = 44100 - len(modified_microphone_data)
                modified_microphone_data = np.array(modified_microphone_data.tolist() + [0 for i in range(number_of_missing_hertz)], dtype = "float32")
        modified_microphone_data = modified_microphone_data[:44100]

        


        

        # Passes an audio sample of an appropriate format into the model for inference
        processed_data_1 = modified_microphone_data
        processed_data_1 = processed_data_1.reshape(input_shape_1)

        processed_data_2 = convert_audio_to_spectrogram(data = modified_microphone_data, hop_length=345*2)
        processed_data_2 = processed_data_2.reshape(input_shape_2)

        processed_data_3 = convert_audio_to_spectrogram(data = modified_microphone_data, hop_length=345)
        processed_data_3 = processed_data_3.reshape(input_shape_3)

        

        
         
        # Performs inference with the instantiated TensorFlow Lite models
        interpreter_1.set_tensor(input_details_1[0]['index'], processed_data_1)
        interpreter_1.invoke()
        probabilities_1 = interpreter_1.get_tensor(output_details_1[0]['index'])
        
        interpreter_2.set_tensor(input_details_2[0]['index'], processed_data_2)
        interpreter_2.invoke()
        probabilities_2 = interpreter_2.get_tensor(output_details_2[0]['index'])
        
        interpreter_3.set_tensor(input_details_3[0]['index'], processed_data_3)
        interpreter_3.invoke()
        probabilities_3 = interpreter_3.get_tensor(output_details_3[0]['index'])

        

        
        # Records which models, if any, identified a gunshot
        model_1_activated = probabilities_1[0][1] >= MODEL_CONFIDENCE_THRESHOLD
        model_2_activated = probabilities_2[0][1] >= MODEL_CONFIDENCE_THRESHOLD
        model_3_activated = probabilities_3[0][1] >= MODEL_CONFIDENCE_THRESHOLD

        # Majority Rules: Determines if a gunshot sound was detected by a majority of the models
        if model_1_activated and model_2_activated or model_2_activated and model_3_activated or model_1_activated and model_3_activated:
            var.set("Gunshot!")
            window.configure(bg="green")
            window.update_idletasks()
        else:
            var.set("Nothing Interesting")
            window.configure(bg="red")
            window.update_idletasks()
        
    # Allows us to capture two seconds of background noise from the microphone for noise reduction
    elif NOISE_REDUCTION_ENABLED and not noise_sample_captured:
        noise_sample = librosa.resample(y = microphone_data, orig_sr = AUDIO_RATE, target_sr = 22050)
        noise_sample = noise_sample[:44100]
        noise_sample_captured = True


