from core import *

def run():
    spectrogram = BasicSpectrogram()
    spectrogram_2 = BasicSpectrogram(x_lim=1000, line_args = {'c': 'red'})
    effects = [spectrogram, spectrogram_2]
    P = Performance(effects)
    P.perform()

if __name__ == "__main__":
    run()