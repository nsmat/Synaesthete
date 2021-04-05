from core import Performance
from effects.spectrogram import BasicSpectrogram, RainbowSpectrogram


def run():
    print('We begin!')
    spectrogram = BasicSpectrogram(x_lim=1000, y_lim=1000, line_args={'c': 'orange'})
    rainbow = RainbowSpectrogram(n_lines=100)

    effects = [rainbow, spectrogram]
    P = Performance(effects, blit=True)
    P.perform()


if __name__ == "__main__":
    run()
