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
import numpy as np
from itertools import cycle

# High level goals
# Performance - even if hefty preloading is required, should run pretty seamlessly.
# Readability - ultimately everything should be classed.
# Flexibility - you should be able to layer effects on as you go/apply multiple effects.

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

    def __init__(self, effects, chunk_size = 2048, form = pyaudio.paInt16,
                 channels = 2, rate = 44100, output_file = None):
        
        # TODO Synaesthete should be passed as an argument. Pointless until it is more configurable.

        # Define Parameters
        self.chunk_size = chunk_size
        self.format = form
        self.channels = channels
        self.rate = rate
        self.output_file = output_file 
        self.interval=1. #milliseconds
        self.starttime = time.time()
        
        # Define Performance Objects
        self.Synaesthete = Synaesthete(chunk_size, effects=effects)
        self.Synaesthete.set_transformer(FourierTransformer(self.chunk_size)) # Classed so we can easily replace if we want

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

    def create_animation(self, background_colour = 'black'):
        """ Create a tk window for plotting in"""

        window = tk.Tk()
        window.configure(background=background_colour)

        fig, ax = plt.subplots()
        
        self.Synaesthete.set_ax(ax)

        print('Opening canvas')

        canvas = FigureCanvasTkAgg(fig, master = window)
        canvas.get_tk_widget().pack(side = "bottom", fill = "none", expand  = "yes")

        print('Animation')

        ani = animation.FuncAnimation(fig, self.Synaesthete.master,
                                        interval=self.interval, blit=False,
                                        frames = 200)

        tk.mainloop()

        self.canvas = canvas
        self.animation = ani

        return    

    def perform(self, printing = True):
        print('Performance Started! Woohoo!')
        self.start_stream()

        # Start Animation
        self.create_animation()
        self.close_stream()
        print('Performance Finished :((((')


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
        self.effects = cycle(effects) # should take a generator functon
        self.ax = None

        # TODO create a switching class that incoporates keys etc
        def switch_function(switch_freq = 0.01):
            return np.random.rand() < switch_freq
        
        self.switcher = switch_function # This should ultimately be a string that points to a switching logic class.

    
    def master(self, frame):
        # TODO make this elegantly handle switching, will need to write two Effects classes to switch between.
        # TODO incorporate button handling
        self.update_data()
        x_data, y_data = self.get_data()
        artists = self.active_effect.get_image(self.get_ax(), x_data, y_data)

        if self.switcher():
            print('Switching to next effect')
            self.ax.clear()
            self.active_effect.init= True
            self.active_effect = self.next_active_effect()

        return artists

    def next_active_effect(self):
        return next(self.effects)

    def update_data(self):
        self._data = self.get_transformed_data()
        
    def get_data(self):
        return self._data
    
    def set_stream(self, stream):
        self._stream = stream

    def get_stream(self):
        return self._stream

    def set_ax(self, ax):
        self.ax = ax
    
    def get_ax(self):
        return self.ax

    def get_data_from_buffer(self):
        stream_read = self.get_stream().read(self.chunk_size, exception_on_overflow=False)
        data = np.frombuffer(stream_read, dtype=np.int16)
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

    The idea here is that they should always take in a common set of arguments, 
        and must have a method:.

    get_image -> returns an iterable of artists that can be used by FuncAnimation

    All other methods/attributes should be used for the animation itself.
    """

    def __init__(self, x_lim = None, y_lim = None, cmap = 'Viridis',
                 line_args = {'drawstyle': 'steps-post', 'c': 'black'}):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.init=True
        self.line_args = line_args
        self.cmap = cmap

    def get_image(self, ax, x_data, y_data):

        if self.init:
            self.line, = ax.plot([], [], **self.line_args)
            self.init = False
        self.line.set_data(x_data, y_data)

        
        if self.y_lim:  
            ylim = self.y_lim
        else:
            ylim = max(y_data)

        ax.set_ylim(0, ylim)
        ax.set_xlim(0, self.x_lim)
        return self.line,







