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

# TODO Chunk size is pretty ugly?
# TODO Deprecate Animator class? Only called by Performance, seems like overkill

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

    def __init__(self, chunk_size = 2048, form = pyaudio.paInt16,
                 channels = 2, rate = 44100, output_file = None):
        
        # TODO Synaesthete should be passed as an argument. Pointless until it is more configurable.

        # Define Parameters
        self.chunk_size = chunk_size
        self.format = form
        self.channels = channels
        self.rate = rate
        self.output_file = output_file 
        
        # Define Performance Objects
        self.Synaesthete = Synaesthete(chunk_size)
        self.Synaesthete.set_transformer(FourierTransformer(self.chunk_size)) # Classed so we can easily replace if we want
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
        self.Synaesthete.set_stream(stream)

        print("Stream Started")

    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.PA.terminate()
        print('Stream Closed')

    def perform(self, printing = True):
        print('Performance Started! Woohoo!')
        self.start_stream()

        # Start Animation
        self.Animator.create_animation()
        self.close_stream()
        print('Performance Finished :((((')


class Animator():
    """Handles Animtion and Plotting Backend"""
    def __init__(self, Synaesthete):
        self.interval=1. #milliseconds
        self.starttime = time.time()
        self.Synaesthete = Synaesthete

    def create_animation(self, background_colour = 'black'):
        """ Create a tk window for plotting in"""

        window = tk.Tk()
        window.configure(background=background_colour)

        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        print('Opening canvas')

        canvas = FigureCanvasTkAgg(fig, master = window)
        canvas.get_tk_widget().pack(side = "bottom", fill = "none", expand  = "yes")

        print('Animation')

        self.Synaesthete.anim_init(ax)

        ani = animation.FuncAnimation(fig, self.Synaesthete.master,
                                        interval=self.interval, blit=False,
                                        frames = 200)

        tk.mainloop()

        self.canvas = canvas
        self.animation = ani

        return

    def plot(self):
        return

class Synaesthete():
    """Where the Magic Happens.

    Ultimately the plan is that there should be a class Effects, which is SubClassed.

    The Effects you want to use are defined at Instantiation 
    
    """
    def __init__(self, chunk_size, transform_type = 'fourier', effects = []):      
        self.data = [], []
        self.stream = None
        self.transformer = None
        self.chunk_size = chunk_size
        
        self.active_effect = effects[0]
        self.effects_generator = (effects[i] for i in range(len(effects))) # check if this works?
        self.effects = effects
     
    
    def master(self, frame):
        # TODO make this elegantly handle switching, will need to write two Effects classes to switch between.
        # TODO incorporate button pressing
        self.update_data()
        x_data, y_data = self.get_data()
        artists = self.active_effect.get_image(x_data, y_data)
        return self.line,

    def next_active_effect(self):
        return next(self.effects_generator) # check if this works?

    def update_data(self):
        self.data = self.get_transformed_data()
        
    def get_data(self):
        return self.data
    
    def set_stream(self, stream):
        self.stream = stream

    def get_stream(self):
        return self.stream

    def get_data_from_buffer(self):
        data = np.frombuffer(self.stream.read(self.chunk_size, exception_on_overflow=False), dtype=np.int16)
        return data
    
    def set_transformer(self, transformer):
        self.transformer = transformer
    
    def get_transformed_data(self):
        data = self.get_data_from_buffer()
        transform_y = self.transformer.transform(data)
        transform_x = self.transformer.frequencies
        return transform_x, transform_y
    

class BasicSpectrogram():
    """First Example of an Effects class.

    The idea here is that they should always take in a common set of arguments, and must have two methods.

    get_image -> returns an iterable of artists that can be used by FuncAnimation
    init_image -> initialises the image? Not 100% sure if needed but will need to check if we can elegantly
                    handle canvas clearing without it
    """


    def __init__(cmaps, x_lim = None, y_lim = None):
        self.cmaps = cmaps
        self.x_lim = None
        self.y_lim = None

    def get_image():
        self.line.set_data(x_data, y_data)
        self.ax.set_ylim(0, max(y_data))
        return line,








