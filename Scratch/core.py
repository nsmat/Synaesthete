import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import moviepy.editor as mpe
from PIL import Image
import time
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pynput import keyboard

# High level goals
# Performance - even if hefty preloading is required, should run pretty seamlessly.
# Readability - ultimately everything should be classed.
# Flexibility - you should be able to layer effects on as you go/apply multiple effects.

# # #  Goals
# 1. Print a livestream of Fourier Coefficients
# 2. Plot a livestream of Fourier Coefficients

class FourierTransformer():
    """ Performs Fourier Transforms"""
    # TODO Check the calculations Zac did.

    def __init__(self, chunk_size):
        # TODO rename variables to be more meaningful
        self.chunk_size = chunk_size
        self.T = 1.0 / (12*chunk_size)
        self.frequencies = np.linspace(0.0, 1.0/(2*self.T), int(chunk_size/2))

    def transform(self, data):
        yf = np.abs(np.fft.fft(data))
        return  2.0/self.chunk_size *np.abs(yf[:self.chunk_size//2]) # TODO Why stop halfway??
        

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

    def get_data_from_buffer(self):
        data = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.int16)
        return data

    def __init__(self, Synaesthete, chunk_size = 1024, form = pyaudio.paInt16,
                 channels = 2, rate = 44100, output_file = None):
        
        # Define Parameters
        self.chunk_size = chunk_size
        self.format = form
        self.channels = channels
        self.rate = rate
        self.output_file = output_file
        
        # Define Performance Objects
        self.Synaesthete = Synaesthete
        self.transformer = FourierTransformer(self.chunk_size) # Classed so we can easily replace if we want
        self.Animator = Animator(self.Synaesthete)

    def start_stream(self):
        PA = pyaudio.PyAudio() # Instantiate the PyAudio instance

        stream = PA.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        self.stream = stream
        self.PA = PA
        self.Synasthete.set_stream(stream)

        print("Stream Started")

    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.PA.terminate()
        print('Stream Closed')

    def get_transformed_data(self):
        data = self.get_data_from_buffer()
        transform_y = self.transformer.transform(data)
        transform_x = self.transformer.frequencies
        return transform_x, transform_y

    def perform(self, printing = True):
        print('Performance Started! Woohoo!')
        self.start_stream()

        # Start Animation
        self.Animator.create_tk_window()

        while True:
            transform_x, transform_y = self.get_transformed_data()
            self.Synaesthete.update_data(transform_x, transform_y)
            if printing:
                print(transform_x)
                print(transform_y)
        self.close_stream()
        print('Performance Finished :((((')


class Animator():
    """Handles Animtion and Plotting Backend"""
    def __init__(self, Synaesthete):
        self.interval=1. #milliseconds
        self.starttime = time.time()
        self.timelength = 10 # seconds
        self.Synaesthete = Synaesthete

    def create_tk_window(self, background_colour = 'black'):
        """ Create a tk window for plotting in"""

        window = tk.Tk()
        window.configure(background=background_colour)

        fig, ax = plt.subplots()

        print('Opening canvas')

        canvas = FigureCanvasTkAgg(fig, master = window)
        canvas.get_tk_widget().pack(side = "bottom", fill = "none", expand  = "yes")

        print('Animation')

        self.Synaesthete.anim_init(ax)

        ani = animation.FuncAnimation(fig, self.Synaesthete.get_image,
                                        interval=self.interval, blit=False)

        tk.mainloop()

        self.canvas = canvas
        self.animation = ani

        return

    def plot(self):
        return

class Synaesthete():
    """Where the Magic Happens"""
    def __init__(self, transform_type = 'fourier', effects = []):      
        self.effects = effects
        self.data = [1, 2, 3,4,5], [1,2,3,4,5]
        self.stream = None
    
    def get_image(self, frame):
        x_data, y_data = self.get_data()
        self.line.set_data(x_data, y_data)    
        return self.line,
    
    def update_data(self, x_data, y_data):
        self.data = x_data, y_data
        

    def get_data(self):
        return self.data

    def anim_init(self, ax):
        line, = ax.plot([], [], lw=2)
        line.set_data([], [])
        self.line = line
         








