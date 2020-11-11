from core import Performance
from effects.spectrogram import BasicSpectrogram, RainbowSpectrogram

def run():
    spectrogram = BasicSpectrogram(x_lim=1000, y_lim = 1000, line_args = {'c': 'orange'})
    rainbow = RainbowSpectrogram(n_lines=20)
    
    effects = [rainbow]
    P = Performance(effects)
    P.perform()

if __name__ == "__main__":
    run()