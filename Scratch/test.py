from core import *

def run(): 
    Syn = Synaesthete() # For now synaesthete does nothing
    P = Performance(Syn)
    P.perform()

if __name__ == "__main__":
    run()