class BasicSpectrogram():
    """First Example of an Effects class.

    The idea here is that they should always take in a common set of arguments (ax and data), 
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