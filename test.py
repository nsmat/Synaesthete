from core import *

def run():
    spectrogram = BasicSpectrogram(x_lim = 1000)
    spectrogram_2 = BasicSpectrogram(x_lim=1000, line_args = {'c': 'red'})

    spectrogram_3 = BasicSpectrogram(x_lim=1000, y_lim = 3000, line_args = {'c': 'orange'})
    
    effects = [spectrogram, spectrogram_2, spectrogram_3]
    P = Performance(effects)
    P.perform()

if __name__ == "__main__":
    run()