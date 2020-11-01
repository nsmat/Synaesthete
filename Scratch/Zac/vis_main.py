
"""
main script for executing audio visualizations. written in python 3.7 (using spyder via anaconda)
"""
import os
import moviepy.editor as mpe
from PIL import Image
import vis_defs as vd
from win32api import GetSystemMetrics
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import tkinter as tk
import pyaudio
import markovify as mfy
import json

dir_path = os.getcwd()
#res = [GetSystemMetrics(0),GetSystemMetrics(1)] #full screen tkinter window
res = [700,500]

#%%
"""
starting image and standard visual size defined here. movies are decomposed into frames for quicker access
"""
visual_size = [600,400]
dpivar = 70

dothis=0
if dothis==1: #can switch this off for speed once its been done once   

    vids_files = []
    vids=[] #holds video duration in seconds, because I'm lazy
    for vid in os.listdir(dir_path+'\\videos'):
        v = mpe.VideoFileClip(dir_path+'\\videos\\'+vid).resize((visual_size[0],visual_size[1]))
        vids_files.append(v)
        vids.append(v.duration)

    count=int(0)
    np.save('vids.npy',vids)    

    for vid in os.listdir(dir_path+'\\videos'):
        v = vids_files[count]
        if not os.path.exists((dir_path+'\\vid_pics\\'+ os.path.splitext(vid)[0])):
            os.mkdir(dir_path+'\\vid_pics\\'+ os.path.splitext(vid)[0])
            for framenum in range(v.reader.nframes):
                frame = v.get_frame((framenum/v.reader.nframes)*v.duration)
                frame = frame.astype(np.uint8)
                imout = Image.fromarray(frame)
                imout.save(dir_path+'\\vid_pics\\'+ os.path.splitext(vid)[0]+'\\'+str(framenum)+'.png')
                
        print(str(count)+' of '+str(len(vids_files))+' completed')
        count+=1
        
    del vids_files
vids = np.load('vids.npy')

#%%
"""
generate markov model for text/ read one from saved json
"""
dothis = 0
if dothis==1:
    textfile = open(dir_path+'\\zac_stories.txt')
    text = textfile.read()
    text_model = mfy.Text(text)
    with open('text_model.json','w') as f:
        json.dump(text_model.to_json(),f)
if dothis==0:
    text_model = mfy.Text.from_json(json.load(open('text_model.json',)))
#%%
"""
main piece executing the visualizer
"""      

#initialize pyaudio stream. right now it takes system sound as mic input-------------------------------
CHUNK = 1024              # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
p=pyaudio.PyAudio()

stream = p.open(
    format=FORMAT, 
    input_device_index=1,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)
#end pyaudio initialize-------------------------------------------------------------------------------------

#initialize various animation stuff
ani_interval=1. #ms
starttime = time.time()
timelength = 10 #s
change = 1
writing=0
double = 0
#deprecated parameters
rad_fact=1  #controls circle growth size, will depend on volume, like maxgridvalue
glitch_fact = 1.5 #if its glitching too much on high volume. use larger # for quieter. try and get inverted images timed for percussion
maxgridvalue=1500
timecheck = time.time()
timecount = 1

#some editable parameters to control effects, esp. based on volume-----------------------------------------
writingchance = 3 #ie 1 in writingchance chance of module having text
printtime = 0 #turn to 1 if you want it to print the average frame time of a module
scale=1 #
#control likelihood of the modules
    #0) grid of notes with colormap
    #1) grid of notes with video
    #2) circles
    #3) glitchy pictures
    #4) video tiling
mod_chance = np.array([2.5,1.4,1.1,1.2,1.2]) #relative likelihood for each 
#mod_chance[3]=100 #turn on to watch specific module
volume = 1

#end editable parameters----------------------------------------------------------------------------------------


#initialize various graphics things------------------------------------------------------------------------------
pcolorfig,pcolorax = plt.subplots(figsize=(visual_size[0]/dpivar,visual_size[1]/dpivar),facecolor='k',dpi=dpivar)
fig = plt.Figure(figsize=(visual_size[0]*scale/dpivar,visual_size[1]*scale/dpivar),facecolor='k',dpi=dpivar)
ax = fig.add_subplot(111) 
ax.axis('off')
#-----------------------------------------------------------------------------------------------------------------

#save out heaps of objects to pass to animate->master-> onward. for convenience.
vis_objects = [res,visual_size,dpivar,ani_interval,pcolorfig,pcolorax, #6 in this row
               starttime,vids,rad_fact,glitch_fact,writingchance,text_model,scale,fig,ax] #15 in 2 rows
sound_objects = [stream,CHUNK,RATE,maxgridvalue,volume]



#master animate function and spacebar ctrl ------------------------------------------------------------------------------
def animate(i,img,vis_objects,sound_objects,mod_chance):
    
    global change
    global starttime
    global timelength
    global timecount
    global timecheck
    timecount = timecount+1

    #call image from master function
    img2 = vd.master(vis_objects,sound_objects,change,mod_chance,starttime,timelength)
    img.set_data(img2)
    
    change = 0 
    if time.time() > starttime+timelength:
        change = 1 
        starttime = time.time()
        timelength = np.random.randint(15,25)
        #print out the average frame time for the module    
        aveframe = (starttime-timecheck)/timecount
        if printtime == 1:
            print(aveframe)
        timecount = 1
        timecheck =starttime
    return img
     

#spacebar to change module
def next_mod(event):
    global change
    global starttime
    global timelength
    global timecount
    global timecheck
     
    change = 1 
    starttime = time.time()
    timelength = np.random.randint(5,20)
    #print out the average frame time for the module    
    aveframe = (starttime-timecheck)/timecount
    if printtime == 1:
        print(aveframe)
    timecount = 1
    timecheck =starttime    
    
def volup(event):
    global sound_objects
    sound_objects[4] = 1.1*sound_objects[4]
    print(sound_objects[4])
def voldown(event):
    global sound_objects
    sound_objects[4] = 0.9*sound_objects[4]
    print(sound_objects[4])
#end animation functions ------------------------------------------------------------------------------------
    
#tk window---------------------------------------------------------------------------------------------------
window = tk.Tk()
window.configure(background='black')
window.geometry(str(res [0])+'x'+str(res[1]))
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack(side = "bottom", fill = "none", expand  = "yes")

img = ax.imshow(np.random.rand(visual_size[1],visual_size[0])) #useful to see code-breaking errors immediately.

#some buttons for basic controls in the visualizer
window.bind('<space>',next_mod) #chance scene on spacebar hit
window.bind('<Down>',voldown) 
window.bind('<Up>',volup)

ani = animation.FuncAnimation(fig, animate, fargs=[img,vis_objects,sound_objects,mod_chance], interval=ani_interval, 
                              blit=False)

tk.mainloop()    
#end tk window-----------------------------------------------------------

#end the pyaudio stream
plt.close(pcolorfig)
stream.stop_stream()
stream.close()
p.terminate()

#ftimecheck = time.time()
#aveframe = (ftimecheck-timecheck)/timecount
#print(aveframe)   

#yf = np.load('yf_test.npy') #for when you want some sample yf data
#xf = np.load('xf_test.npy') #for when you want some sample xf data

#%%
#get input device index
#p = pyaudio.PyAudio()
#info = p.get_host_api_info_by_index(0)
#numdevices = info.get('deviceCount')
#for i in range(0, numdevices):
#        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

        
        