import pyaudio
import matplotlib.pyplot as pyplot
import numpy as np
import tkinter as tk
import moviepy.editor as mpe
from PIL import Image
import time
import matplotlib.animation as animation

# High level goals
# Performance - even if hefty preloading is required, should run pretty seamlessly.
# Readability - ultimately everything should be classed.
# Flexibility - you should be able to layer effects on as you go/apply multiple effects.

# # #  Goals
# 1. Print a livestream of Fourier Coefficients
# 2. 


class FourierTransformer():
    """ Performs Fourier Transforms"""

    # TODO Check the calculations Zac did.

    def __init__(self, chunk_size):
        self.T = 1.0 / (12*chunk_size)
        self.xf = np.linspace(0.0, 1.0/(2*self.T), int(chunk_size/2))

    def transform(self, data, chunk_size, volume):
        yf = np.abs(np.fft.fft(data))*volume
        return  2.0/chunk_size *np.abs(yf[:chunk_size//2]) # TODO Why stop halfway??
        

class Performance():
    """Handles PyAudio inputs.

    Attributes:
        - chunk: The number of samples that are bussed from the input through processing at one time.
        - format - the pyaudio input format.
        - channels: the number
        - rate: The fixed sampling rate per second. 
        - record_seconds: Number of seconds to activate/deactivate the stream for.

    Methods:
        ....

    """

    @staticmethod
    def get_data_from_buffer(self, stream):
        data = np.frombuffer(stream.read(self.chunk_size, dtype=np.int16))
        return data

    def __init__(self, Synaesthete, chunk_size = 1024, form = pyaudio.paInt16,
                 channels = 2, rate = 44100, output_file = None):
        
        self.chunk_size = chunk_size
        self.format = form
        self.channels = channels
        self.rate = rate
        self.output_file = output_file
        self.Synaesthete = Synaesthete

    def start_stream(self):
        PA = pyaudio.PyAudio() # Instantiate the PyAudio instance

        stream = PA.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        self.stream = stream
        self.PA = PA

        print("Stream Started")

    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.PA.terminate()
        print('Stream Closed')

class Animate():
    def __init__():
        self.interval=1. #ms
        self.starttime = time.time()
        self.timelength = 10 #s

    def create_tk_window(self):
        """ Create a tk window for plotting in"""

        window = tk.Tk()
        window.configure(background='black')
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(side = "bottom", fill = "none", expand  = "yes")

        ani = animation.FuncAnimation(fig, animate, interval=ani_interval, blit=False)

        tk.mainloop()    
        return

    def plot(self):
        return

class Synaesthete():
    def __init__(self, chunk_size, transform_type, effects = []):
        self.transform_type = transform_type

        if self.transform_type == 'fourier':
            self.transform  = FourierTransformer(chunk_size)
        else:
            raise NotImplementedError('Only fourier transforms are available. Write more code!')
        
        self.effects = effects

    def visualise(self, data):
        return 




