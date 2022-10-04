#!/usr/bin/env python
# coding: utf-8

# Package Imports #
 
import os
import pyaudio
import sys
import librosa
import logging
import time
import schedule
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


# Configuring the Logger #

logger = logging.getLogger('debugger')
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('output.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)




window = Tk()
window.configure(bg="red")
var = StringVar()
soundclass = Label(window, textvariable=var)
soundclass.config(font=("Courier", 90))

soundclass.pack(padx=200, pady=400)


# Variable Initializations #

AUDIO_FORMAT = pyaudio.paFloat32
AUDIO_RATE = 44100
NUMBER_OF_AUDIO_CHANNELS = 1
AUDIO_DEVICE_INDEX = 1
NUMBER_OF_FRAMES_PER_BUFFER = 4410
SAMPLE_DURATION = 2
AUDIO_VOLUME_THRESHOLD = 0.01
NOISE_REDUCTION_ENABLED = False
MODEL_CONFIDENCE_THRESHOLD = 0.5
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = AUDIO_RATE // 2
NUMBER_OF_MELS = 128
NUMBER_OF_FFTS = NUMBER_OF_MELS * 20
ALERT_MESSAGE = "ALERT: A Gunshot Was Detected on "
NETWORK_COVERAGE_TIMEOUT = 3600
DESIGNATED_ALERT_RECIPIENTS = []
SCHEDULED_LOG_FILE_TRUNCATION_TIME = "00:00"
sound_data = np.zeros(0, dtype = "float32")
noise_sample_captured = False
gunshot_sound_counter = 1
noise_sample = []
audio_analysis_queue = Queue()

DIRPATH = "/home/user/Downloads/raspberry_pi/"


# Loading in Augmented Labels #

labels = np.load(os.path.join(DIRPATH, "augmented_labels.npy"))


# Binarizing Labels #

labels = np.array([("gun_shot" if label == "gun_shot" else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))


## Librosa Wrapper Function Definitions ##

def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.logamplitude(x, ref_power=1.0, amin=1e-20, top_db=80.0)  # Librosa 0.4.2 functionality


def _db_to_amp(x):
    return librosa.core.perceptual_weighting(x, frequencies=1.0)  # Librosa 0.4.2 functionality


# Custom Noise Reduction Function Definition #

def remove_noise(audio_clip,
                 noise_clip,
                 n_grad_freq=2,
                 n_grad_time=4,
                 n_fft=2048,
                 win_length=2048,
                 hop_length=512,
                 n_std_thresh=1.5,
                 prop_decrease=1.0,
                 verbose=False):
    """ Removes noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        verbose: Whether to display time statistics for the noise reduction process

    Returns:
        array: The recovered signal with noise subtracted

    """

    # Debugging
    if verbose:
        start = time.time()

    # Takes a STFT over the noise sample
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # Converts the sample units to dB

    # Calculates statistics over the noise sample
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    # Debugging
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()

    # Takes a STFT over the signal sample
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))

    # Debugging
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()

    # Calculates value to which to mask dB
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))

    # Debugging
    if verbose:
        print("Noise Threshold & Mask Gain in dB: ", noise_thresh, mask_gain_dB)

    # Creates a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1]
    )

    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

    # Calculates the threshold for each frequency/time bin
    db_thresh = np.repeat(np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
                          np.shape(sig_stft_db)[1],
                          axis=0).T

    # Masks segment if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh

    # Debugging
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()

    # Convolves the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    # Debugging
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()

    # Masks the signal
    sig_stft_db_masked = (sig_stft_db * (1 - sig_mask)
                          + np.ones(np.shape(mask_gain_dB))
                          * mask_gain_dB * sig_mask)  # Masks real

    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)

    # Debugging
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()

    # Recovers the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )

    # Debugging
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))

    # Returns noise-reduced audio sample
    return recovered_signal


# Converting 1D Sound Arrays into Spectrograms #

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)
    if amin <= 0:
        logger.debug("ParameterError: amin must be strictly positive")
    if np.issubdtype(S.dtype, np.complexfloating):
        logger.debug("Warning: power_to_db was called on complex input so phase information will be discarded.")
        magnitude = np.abs(S)
    else:
        magnitude = S
    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    if top_db is not None:
        if top_db < 0:
            logger.debug("ParameterError: top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def convert_audio_to_spectrogram(data):
    spectrogram = librosa.feature.melspectrogram(y=data, sr=AUDIO_RATE,
                                                 hop_length=HOP_LENGTH,
                                                 fmin=MINIMUM_FREQUENCY,
                                                 fmax=MAXIMUM_FREQUENCY,
                                                 n_mels=NUMBER_OF_MELS,
                                                 n_fft=NUMBER_OF_FFTS)
    spectrogram = power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


# WAV File Composition Function #

# Saves a two-second gunshot sample as a WAV file
def create_gunshot_wav_file(microphone_data, index, timestamp):
    pass

    
# Log File Truncation Function #
        
def clear_log_file():
    with open("output.log", 'w'):
        pass


# ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K" #

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


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


### --- ###

# Multithreaded Inference: A callback thread adds two-second samples of microphone data to an audio analysis
# queue; the main thread, an audio analysis thread, detects the presence of gunshot sounds in samples retrieved from
# the audio analysis queue.

### --- ###


# Defining Threads #



## Callback Thread ##

def callback(in_data, frame_count, time_info, status):
    global sound_data
    sound_buffer = np.frombuffer(in_data, dtype="float32")
    sound_data = np.append(sound_data, sound_buffer)
    if len(sound_data) >= 88200:
        audio_analysis_queue.put(sound_data)
        current_time = time.ctime(time.time())
        audio_analysis_queue.put(current_time)
        sound_data = np.zeros(0, dtype="float32")
    return sound_buffer, pyaudio.paContinue



form_1 = pyaudio.paFloat32 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 3 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)


pa = pyaudio.PyAudio()

stream  = pa.open(format = form_1,
                    rate = samp_rate,
                    channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk, 
                    stream_callback=callback)


# Starts the callback thread
stream.start_stream()
logger.debug("--- Listening to Audio Stream ---")


### Audio Analysis Thread

# Starts the scheduler for clearing the primary log file
schedule.every().day.at(SCHEDULED_LOG_FILE_TRUNCATION_TIME).do(clear_log_file)

# This thread will run indefinitely
while True:
    # Refreshes the scheduler
    schedule.run_pending()
    
    # Gets a sample and its timestamp from the audio analysis queue
    microphone_data = np.array(audio_analysis_queue.get(), dtype = "float32")
    time_of_sample_occurrence = audio_analysis_queue.get()
    
    # Cleans up the global NumPy audio data source
    sound_data = np.zeros(0, dtype = "float32")
        
    # Finds the current sample's maximum frequency value
    maximum_frequency_value = np.max(microphone_data)
        
    # Determines whether a given sample potentially contains a gunshot
    if maximum_frequency_value >= AUDIO_VOLUME_THRESHOLD:
        


        # Displays the current sample's maximum frequency value
        logger.debug("The maximum frequency value of a given sample before processing: " + str(maximum_frequency_value))
        
        # Post-processes the microphone data
        modified_microphone_data = librosa.resample(y = microphone_data, orig_sr = AUDIO_RATE, target_sr = 22050)
        if NOISE_REDUCTION_ENABLED and noise_sample_captured:
                # Acts as a substitute for normalization
                modified_microphone_data = remove_noise(audio_clip = modified_microphone_data, noise_clip = noise_sample)
                number_of_missing_hertz = 44100 - len(modified_microphone_data)
                modified_microphone_data = np.array(modified_microphone_data.tolist() + [0 for i in range(number_of_missing_hertz)], dtype = "float32")
        modified_microphone_data = modified_microphone_data[:44100]

        # Passes an audio sample of an appropriate format into the model for inference
        processed_data_1 = modified_microphone_data
        processed_data_1 = processed_data_1.reshape(input_shape_1)

        HOP_LENGTH = 345 * 2
        processed_data_2 = convert_audio_to_spectrogram(data = modified_microphone_data)
        processed_data_2 = processed_data_2.reshape(input_shape_2)
            
        HOP_LENGTH = 345
        processed_data_3 = convert_audio_to_spectrogram(data = modified_microphone_data)
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

        pred1 = label_binarizer.inverse_transform(probabilities_1[:, 0])[0]
        pred2 = label_binarizer.inverse_transform(probabilities_2[:, 0])[0]
        pred3 = label_binarizer.inverse_transform(probabilities_3[:, 0])[0]

        

        preds = [pred1, pred2, pred3]
        count = 0
        for pred in preds:
            if pred != "other":
                count += 1

        
        var.set(str("{}".format("Gunshot!" if count > 1 else "Nothing Interesting")))
        window.configure(bg= "green" if count > 1 else "red")
        window.update_idletasks()
        
        # Records which models, if any, identified a gunshot
        model_1_activated = probabilities_1[0][1] >= MODEL_CONFIDENCE_THRESHOLD
        model_2_activated = probabilities_2[0][1] >= MODEL_CONFIDENCE_THRESHOLD
        model_3_activated = probabilities_3[0][1] >= MODEL_CONFIDENCE_THRESHOLD

        # Majority Rules: Determines if a gunshot sound was detected by a majority of the models
        if model_1_activated and model_2_activated or model_2_activated and model_3_activated or model_1_activated and model_3_activated:
                

            # Makes a WAV file of the gunshot sample
            create_gunshot_wav_file(modified_microphone_data, gunshot_sound_counter, time_of_sample_occurrence)

            # Increments the counter for gunshot sound file names
            gunshot_sound_counter += 1
    
    # Allows us to capture two seconds of background noise from the microphone for noise reduction
    elif NOISE_REDUCTION_ENABLED and not noise_sample_captured:
        noise_sample = librosa.resample(y = microphone_data, orig_sr = AUDIO_RATE, target_sr = 22050)
        noise_sample = noise_sample[:44100]
        noise_sample_captured = True


