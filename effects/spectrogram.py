import numpy as np
from itertools import cycle
from matplotlib import cm

# TODO create parent class. Requirements are init_func, get_image and plotting_func.

class BasicSpectrogram():
    """First Example of an Effects class.

    The idea here is that they should always take in a common set of arguments (ax and data), 
        and must have a method:.

    get_image -> returns an iterable of artists that can be used by FuncAnimation

    All other methods/attributes should be used for the animation itself.
    """

    def __init__(self, x_lim = None, y_lim = None, cmap = 'jet',
                 line_args = {'drawstyle': 'steps-post', 'c': 'black'}):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.init=True
        self.line_args = line_args
        self.cmap = cmap

    def get_image(self, ax, x_data, y_data):

        self.init_func(ax)

        self.line.set_data(x_data, y_data)

        if self.y_lim:  
            ylim = self.y_lim
        else:
            ylim = max(y_data)

        ax.set_ylim(0, ylim)
        ax.set_xlim(0, self.x_lim)
        return self.line,

    def init_func(self, ax):
        if self.init:
            self.line, = ax.plot([], [], **self.line_args)
            self.init = False

class RainbowSpectrogram():

    # TODO add ability to plot based on specific frequencies
    def __init__(self, cmap_name = 'jet', n_lines = 20, y_func = np.sin):
        self.cmap = cm.get_cmap(cmap_name)
        self.x = np.linspace(0, 4*np.pi, 10000)
        self.y_func = y_func
        self.init = True
        self.n_lines = n_lines
        
        self.set_colours()

    def set_colours(self):
        self.colours =  self.cmap(np.linspace(0,1, self.n_lines))
        
    def get_image(self, ax, x_data, y_data):
        volume = y_data.sum()//100

        self.init_func(ax)

        for i in range(self.n_lines):
            line = self.lines[i]
            line.set_data(self.x, self.y_func(self.x)*i*volume)
        
        ax.set_xlim(0, max(self.x))
        ax.set_ylim(-20*self.n_lines, 20*self.n_lines)

        return self.lines

    def init_func(self,ax):
        if self.init:
            self.lines = []
            for i in range(self.n_lines):
                line = ax.plot([], [], color=self.colours[i])
                self.lines += line
            self.init = False

