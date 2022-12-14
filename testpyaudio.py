import pyaudio

p = pyaudio.PyAudio()

for ii in range(p.get_device_count()):
    device = p.get_device_info_by_index(ii)
    print("Name: ", device.get("name"), "Index: ", ii, "# Input Channels: " , device["maxInputChannels"], "Sample Rate: ", device["defaultSampleRate"])


stream = p.open(format=pyaudio.paFloat32,
                 rate=44100,
                 channels=1,
                 input_device_index=1,
                 frames_per_buffer=512,
                 input=True)




# import pyaudio
# import wave

# form_1 = pyaudio.paInt16 # 16-bit resolution
# chans = 1 # 1 channel
# samp_rate = 44100 # 44.1kHz sampling rate
# chunk = 4096 # 2^12 samples for buffer
# record_secs = 3 # seconds to record
# dev_index = 2 # device index found by p.get_device_info_by_index(ii)
# wav_output_filename = 'test1.wav' # name of .wav file

# audio = pyaudio.PyAudio() # create pyaudio instantiation

# # create pyaudio stream
# stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
#                     input_device_index = dev_index,input = True, \
#                     frames_per_buffer=chunk)
# print("recording")
# frames = []

# # loop through stream and append audio chunks to frame array
# for ii in range(0,int((samp_rate/chunk)*record_secs)):
#     data = stream.read(chunk)
#     frames.append(data)

# print("finished recording")

# # stop the stream, close it, and terminate the pyaudio instantiation
# stream.stop_stream()
# stream.close()
# audio.terminate()

# # save the audio frames as .wav file
# wavefile = wave.open(wav_output_filename,'wb')
# wavefile.setnchannels(chans)
# wavefile.setsampwidth(audio.get_sample_size(form_1))
# wavefile.setframerate(samp_rate)
# wavefile.writeframes(b''.join(frames))
# wavefile.close()
