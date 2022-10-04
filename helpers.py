import librosa
import numpy as np
import six
from constants import *
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


def convert_audio_to_spectrogram(data, hop_length):
    spectrogram = librosa.feature.melspectrogram(y=data, sr=AUDIO_RATE,
                                                 hop_length=hop_length,
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
