#live music modules to be imported into visualizations.py

#i'll make these maximally efficient later on... if I need to

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import skimage.io as io
from skimage.transform import rescale
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as PathEffects

dir_path = os.getcwd()

#%%
#master function for selecting module with relevant parameters
    
def master(vis_objects,sound_objects,change,mod_chance,starttime,timelength):
   
    maxgridvalue = sound_objects[3]
    writingchance = vis_objects[10]
    text_model = vis_objects[11]
    visual_size = vis_objects[1]

    noctaves=9
    ntemperament=12
    full_cmap_list = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 
                      'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 
                      'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 
                      'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
                      'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 
                      'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 
                      'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
                      'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2',
                      'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 
                      'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
                      'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 
                      'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 
                      'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 
                      'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r',
                      'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 
                      'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
                      'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 
                      'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 
                      'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 
                      'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                      'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                      'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted',
                      'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
   
    #various randomized parameters which need to be stored globally ------
    global colormap
    global glitch
    global video
    global videotype
    global randmod
    global circletype
    global cmap_shift
    global cmap_colors
    global radius
    global c_swap
    global pic
    global writing
    global sentence
    global stack
    global loc
    global angle
    global double
    global copynum
    global sizes
    global xlocs
    global ylocs

    #-----------------------------------------------------------------------
    
    #setting up randomized parameters
    mod_chance= np.cumsum(mod_chance/np.sum(mod_chance))
    
    if change ==1:
        randmod = np.random.rand()
        writing = np.random.randint(writingchance)
        double = 0
        if np.random.rand()>.8:
            double = 1    
        stack = np.random.randint(3) #1 if the words are on top of each other and 0 if spread out. could write other options
        sentence = text_model.make_short_sentence(150)
#        print(sentence)
        if stack == 0:
            xloc = np.random.rand(len(sentence.split()))*.8 +.1
            yloc = 1- (np.sort(np.random.rand(len(sentence.split())))*.7 +.15)
        if stack == 1:
            xloc = np.zeros(len(sentence.split())) + np.random.rand()*.7 +.15 + (((np.linspace(0,1,num=len(sentence.split())))*.1))
            if np.random.rand()>0.5:
                xloc = np.zeros(len(sentence.split())) + np.random.rand()*.7 +.15 + (((1-np.linspace(0,1,num=len(sentence.split())))*.1))
            yloc = np.zeros(len(sentence.split())) + np.random.rand()*.7 +.15 + (((1-np.linspace(0,1,num=len(sentence.split())))*.1))
#            xloc = np.zeros(len(sentence.split())) + np.random.rand()*.7 +.15
#            yloc = np.zeros(len(sentence.split())) + np.random.rand()*.7 +.15
        if stack == 2:
            xloc = np.zeros(len(sentence.split())) + np.random.rand()*.7 +.15
            yloc = 1- ((np.linspace(0,1,num=len(sentence.split())))*.7 +.15)


        loc = [xloc,yloc]
    writing_objects = [writing,change,starttime,sentence,timelength,loc,randmod,mod_chance,double]
    #----------------------------------------------------------------------------
    
    
    #module list -----------------------------------------------------------------------------------
    
    #1: basic notegrid module, magma .........................................
    if randmod < mod_chance[0]:
        notecloseparam=0.01
        
        if change == 1:
            colormap = full_cmap_list[np.random.randint(len(full_cmap_list))]
            glitch=0
            if np.random.rand()>0.6:
                glitch=1
                
        img2 = notegrid(vis_objects,sound_objects,noctaves,ntemperament,
                                           maxgridvalue,notecloseparam,colormap,glitch,writing_objects)
        
        #randomly overlay glitch
        if np.random.rand()>0.9:
            img2=np.floor(img2*1.)
            invert=0
            img2 = picglitch(vis_objects,sound_objects,img2,invert)        
        
    #2: notegrid with video....................................................
    if mod_chance[0] < randmod <= mod_chance[1]:
        notecloseparam=0.01
        colormap = 'binary_r'
        if change == 1:
            glitch=0
            if np.random.rand()>0.75:
                glitch=1
            video = np.random.randint(len(os.listdir(dir_path+'\\videos')))
            videotype = 0
            if np.random.rand()>0.75:
                videotype = 1 #this one does the grid subtractively instead

        img2 = notegrid_video(vis_objects,sound_objects,noctaves,ntemperament,
                                           notecloseparam,maxgridvalue,colormap,video,
                                           videotype,glitch,writing_objects)
        #randomly overlay glitch
        if np.random.rand()>0.8:
            img2=np.floor(img2*255)
            img2.astype(np.int)
            invert=0
            img2 = picglitch(vis_objects,sound_objects,img2,invert)

           
    #3: circles.................................................................
    if mod_chance[1] < randmod <= mod_chance[2]:
        if change == 1:
            circletype = np.random.randint(2) #single amplitude, 3 circle freq, triangle
            colormap = full_cmap_list[np.random.randint(len(full_cmap_list))]
#            cmap_shift = np.random.randint(2)
            cmap_shift = 1
            cmap_colors = np.random.rand(6)
            radius = 1
            c_swap = 1
            angle = np.random.rand()*np.pi/2

        img2 = circles(vis_objects,sound_objects,circletype,colormap,cmap_shift,writing_objects,angle)
        
        #randomly overlay glitch
        if np.random.rand()>0.65:
            img2=np.floor(img2*1.)
            invert=1
            img2 = picglitch(vis_objects,sound_objects,img2,invert)   

    #4: picglitch...............................................................
    if mod_chance[2] < randmod <= mod_chance[3]:
        if change ==1 :
            picname = np.random.choice(os.listdir(dir_path+'\\smallerpics'))
            pic =  io.imread(dir_path+'\\smallerpics\\'+picname)
        invert = 1
        img2 = picglitch(vis_objects,sound_objects,pic,invert)
            
#        if writing == 0:
#            markovtext(vis_objects,writing_objects)    

        plt.cla()
        
    #5: vidtiles.............................................................................
    if mod_chance[3] < randmod <= mod_chance[4]:
        if change == 1:
            video = np.random.randint(len(os.listdir(dir_path+'\\videos')))
            copynum = np.random.randint(10,20)
            sizes = np.random.randint(1,3,copynum)
            xlocs = (np.random.rand(copynum)-0.5)*2*visual_size[0] #bottom left corner location, can be outside of shot though
            ylocs = (np.random.rand(copynum)-0.5)*2*visual_size[1]
            xlocs = xlocs.astype(np.int)
            ylocs = ylocs.astype(np.int)
        img2 = vidtile(vis_objects,sound_objects,video,writing_objects,copynum,sizes,xlocs,ylocs)
        #randomly overlay glitch
        if np.random.rand()>0.5:
            img2=np.floor(img2*255)
            img2.astype(np.int)
            invert=1
            img2 = picglitch(vis_objects,sound_objects,img2,invert)

    #end module list ---------------------------------------------------------------------------
    
    #apply markov writing on top of the image            
#    if writing == 0:
#        markovtext(vis_objects,change,starttime,sentence,timelength,loc)
    
    return img2

#%%
#turning matplotlib figures into numpy arrays for quicker saving, etc. stole this from somewhere.
#they got the w,h confused and its very annoying so I've got the _trans version to do it regularly
        
def fig2data ( fig ):

    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
def fig2data_trans ( fig ):

    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    h,w = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

#%%
#notegrid
#basic heatmap of notes being played
    
def notegrid(vis_objects,sound_objects,noctaves,ntemperament,maxgridvalue,
             notecloseparam,colormap,glitch,writing_objects):
    #importing objects
    volume = sound_objects[4]
    
#    visual_size = vis_objects[1]
    stream=sound_objects[0]
    CHUNK=sound_objects[1]
    pcolorfig = vis_objects[4]
    pcolorax = vis_objects[5]
    
    #initializing stuff
    notesgrid = np.flip(np.reshape(np.arange(noctaves*ntemperament),[noctaves,ntemperament]),axis=0)
    freqgrid = np.round((2**((notesgrid+1-49)/ntemperament))*440) #Hz
    dispgrid = np.zeros([noctaves,ntemperament])
    
    #fourier transform, I hope this is correct lol
    T = 1.0 / (12*CHUNK)
    data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
    yf = np.abs(np.fft.fft(data))*volume
    yf = 2.0/CHUNK *np.abs(yf[:CHUNK//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), int(CHUNK/2))

#    new grid updating method to be quicker
    for j in range(noctaves*ntemperament):
        grid_indices = np.argwhere(notesgrid==j)[0]
        note = freqgrid[grid_indices[0],grid_indices[1]]
        upperlimit_ind = (np.abs(xf - note*(1+notecloseparam))).argmin()
        lowerlimit_ind = (np.abs(xf - note*(1-notecloseparam))).argmin()
        max_volume = np.max(yf[lowerlimit_ind:upperlimit_ind+1])*((((noctaves-grid_indices[0]))/noctaves)+1) 
        dispgrid[grid_indices[0],grid_indices[1]]=max_volume

    dispgrid = np.ceil((np.floor((maxgridvalue/(dispgrid+1))))/(np.floor((maxgridvalue/(dispgrid+1)))+1))*dispgrid + np.floor(dispgrid/maxgridvalue)*maxgridvalue
      
#    t1 = time.time()

    #figure generating and saving
    if glitch == 0:
        pcolorax.pcolormesh(dispgrid, edgecolors='k', linewidths=1,vmin = 0, vmax = maxgridvalue,cmap=colormap)
        pcolorax.invert_yaxis()
        pcolorax.axis('off')
        img=fig2data_trans(pcolorfig)
        plt.cla()
#        img = np.random.rand(visual_size[1],visual_size[0])


#        plt.close(pcolorfig)
    if glitch == 1:
        pcolorax.pcolormesh(dispgrid, edgecolors='k', linewidths=1,vmin = 0, vmax = maxgridvalue,cmap=colormap)
        pcolorax.invert_yaxis()
        pcolorax.axis('off')
        img = fig2data(pcolorfig)
        plt.cla()
#        img = np.random.rand(visual_size[1],visual_size[0])

    writing = writing_objects[0]
    if writing == 0:
        markovtext(vis_objects,writing_objects)    

    return img        
        
#%%
#notegrid_video
#subtractive version of notegrid but with a video for the background
        
def notegrid_video(vis_objects,sound_objects,noctaves,ntemperament,notecloseparam,maxgridvalue,colormap,
                   video,videotype,glitch,writing_objects):
    
    stream=sound_objects[0]
    CHUNK=sound_objects[1]
    volume = sound_objects[4]    
    pcolorfig = vis_objects[4]
    pcolorax = vis_objects[5]
    starttime = vis_objects[6]
    vids = vis_objects[7]

    notesgrid = np.flip(np.reshape(np.arange(noctaves*ntemperament),[noctaves,ntemperament]),axis=0)
    freqgrid = np.round((2**((notesgrid+1-49)/ntemperament))*440) #Hz
    dispgrid = np.zeros([noctaves,ntemperament])
    
    #fourier transform, I hope this is correct lol
    T = 1.0 / (12*CHUNK)
    data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
    yf = np.abs(np.fft.fft(data))*volume
    yf = 2.0/CHUNK *np.abs(yf[:CHUNK//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), int(CHUNK/2))

    #new grid updating method to be quicker
    for j in range(noctaves*ntemperament):
        grid_indices = np.argwhere(notesgrid==j)[0]
        note = freqgrid[grid_indices[0],grid_indices[1]]
        upperlimit_ind = (np.abs(xf - note*(1+notecloseparam))).argmin()
        lowerlimit_ind = (np.abs(xf - note*(1-notecloseparam))).argmin()
        #make higher notes a bit brighter
        max_volume = np.max(yf[lowerlimit_ind:upperlimit_ind+1])*((((noctaves-grid_indices[0]))/noctaves)+1)
        dispgrid[grid_indices[0],grid_indices[1]]=max_volume
        
    #prevent static
    for i in range(noctaves*ntemperament):
        grid_indices = np.argwhere(notesgrid==j)[0]
        if dispgrid[grid_indices[0],grid_indices[1]] < np.max(dispgrid)/(2.) :
            dispgrid[grid_indices[0],grid_indices[1]]=0
        if dispgrid[grid_indices[0],grid_indices[1]] < 100:
            dispgrid[grid_indices[0],grid_indices[1]]=0
   
#    t1 = time.time()
    #figure generating and saving
    if videotype == 0 and glitch == 0:
        pcolorax.pcolormesh(dispgrid, edgecolors='k', linewidths=1,vmin = 0, vmax = maxgridvalue,cmap='binary_r')
    if videotype ==1 :
        dispgrid = np.max(dispgrid)-dispgrid
        pcolorax.pcolormesh(dispgrid, edgecolors='k', linewidths=1,cmap='binary_r')
    if videotype == 0 and glitch ==1 :
        pcolorax.pcolormesh(dispgrid, edgecolors='k', linewidths=1,cmap='binary_r')       
    pcolorax.invert_yaxis()
    pcolorax.axis('off')
    img = fig2data_trans(pcolorfig)[:,:,0:3]
    plt.cla()

    #video frame stuff
    t = time.time()-starttime
    framet = t%vids[video] #loop if video is too short
    vid_name = os.path.splitext(os.listdir(dir_path+'\\videos')[video])[0]
    nframes=len(os.listdir(dir_path+'\\vid_pics\\'+vid_name))
    framenum = int((framet/vids[video])*nframes)
    if glitch==1:
        framepic = io.imread(dir_path+'\\vid_pics\\'+vid_name+'\\'+str(framenum)+'.png')
    if glitch==0:
        framepic = io.imread(dir_path+'\\vid_pics\\'+vid_name+'\\'+str(framenum)+'.png')
        framepic=framepic.astype(np.float32)/(255.)
    
    finalpic = img*framepic/(255.)
    
    writing = writing_objects[0]
    if writing == 0:
        markovtext(vis_objects,writing_objects)    

#    print(time.time()-t1)
    return finalpic
#%%
#shift the global cmap list. output the colormap itself
    
def color_shift(version):
    global cmap_colors #for simple two-color colormap between 0 and 1
    global pm_factors
    
    if version == 1: 
        factor = 0.05
        pm_factors = (-1)**np.round(np.random.rand(6))
        cmap_colors = cmap_colors + pm_factors*factor*np.random.rand(6)
        
        for j in range(len(cmap_colors)):
            if cmap_colors[j] > 1:
                cmap_colors[j] = cmap_colors[j] - factor
                pm_factors[j]=pm_factors[j]*(-1)
            elif cmap_colors[j] < 0:
                cmap_colors[j]=cmap_colors[j] + factor
                pm_factors[j]=pm_factors[j]*(-1)                
        cmap = ListedColormap([(cmap_colors[0],cmap_colors[1],cmap_colors[2]),(cmap_colors[3],cmap_colors[4],cmap_colors[5])])
    return cmap
#%%
#circles
    #circles expanding ?
def circles(vis_objects,sound_objects,circletype,colormap,cmap_shift,writing_objects,angle):
    
    stream=sound_objects[0]
    CHUNK=sound_objects[1]
    visual_size = vis_objects[1]
    pcolorfig = vis_objects[4]
    pcolorax = vis_objects[5]
    rad_fact = vis_objects[8]
    volume = sound_objects[4]
    global radius
    global c_swap
       
#    fourier transform
    T = 1.0 / (12*CHUNK)
    data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
    yf = np.abs(np.fft.fft(data))*volume
    yf = 2.0/CHUNK *np.abs(yf[:CHUNK//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), int(CHUNK/2))
    
    bass_ind = next(x[0] for x in enumerate(xf) if x[1] > 250) #Hz hopefully
    mid_ind = next(x[0] for x in enumerate(xf) if x[1] > 2000)
    
    if cmap_shift == 1:
        colormap = color_shift(1)                
    
    if circletype == 0:
        #circles expanding
        xx, yy = np.mgrid[:visual_size[1], :visual_size[0]]
        fact = .99 #maps yf maximum to radius
        circle = (xx - int(visual_size[1]/2)) ** 2 + (yy -  int(visual_size[0]/2)) ** 2
        
        #update radius
        updown = ((-1)**(np.random.randint(3)))
        radius = (radius) + updown*np.max(data)*rad_fact*(1 + ((1+updown)/2)*0.01*((radius+1)/(visual_size[1]/2)))      
        
        if radius > ((visual_size[0]/2)**2)+((visual_size[1]/2)**2):
            radius = 1
            c_swap = c_swap*(-1)
        if radius < 0:
            radius = 1
        if c_swap == 1:
            disk = (circle < radius)
        if c_swap == -1:
            disk = (circle > radius)
        
        pcolorax.imshow(disk, cmap=colormap)
        pcolorax.axis('off')
        img = fig2data_trans(pcolorfig)[:,:,0:3]
        plt.cla()
      
    if circletype == 1:
        #circle drawing, 3 circles in different parts of frequency
        xx, yy = np.mgrid[:visual_size[1], :visual_size[0]]
        fact = .99 #maps yf maximum to radius
        circle0 = (xx - int(visual_size[1]/2)) ** 2 + (yy -  int(visual_size[0]/4)) ** 2
        circle1 = (xx - int(visual_size[1]/2)) ** 2 + (yy -  int(visual_size[0]/2)) ** 2
        circle2 = (xx - int(visual_size[1]/2)) ** 2 + (yy -  int(visual_size[0]*3/4)) ** 2
        disk0 = (circle0 < (np.max(yf[0:bass_ind]*3)*(fact**2)*rad_fact))
        disk1 = (circle1 < (np.max(yf[bass_ind:mid_ind]*5)*(fact**2)*rad_fact))
        disk2 = (circle2 < (np.max(yf[mid_ind:-1]*10)*(fact**2)*rad_fact))
        pcolorax.imshow((disk0+disk1+disk2)/3, cmap=colormap)
        pcolorax.axis('off')
        img = fig2data_trans(pcolorfig)[:,:,0:3]
        plt.cla()
    
    if circletype == 2:
        #triangles with pointy ends lol. this just doesn't work well- its slow and weird. could fix another time.
        fact = 0.99
        xx,yy = np.mgrid[:visual_size[1], :visual_size[0]]
        bassrad = np.sqrt(np.max(yf[0:bass_ind]*3)*(fact**2))*rad_fact*3
        midrad = np.sqrt(np.max(yf[bass_ind:mid_ind]*5)*(fact**2))*rad_fact*3
        trebrad = np.sqrt(np.max(yf[mid_ind:-1]*10)*(fact**2))*rad_fact*3

        bp = np.array([bassrad*np.cos(angle) +visual_size[0]/2,bassrad*np.sin(angle) +visual_size[1]/2]).astype(np.int)
        mp = np.array([midrad*np.cos(angle+ np.pi/3) +visual_size[0]/2,midrad*np.sin(angle+ np.pi/3) +visual_size[1]/2]).astype(np.int)
        tp = np.array([trebrad*np.cos(angle+ 2*np.pi/3) +visual_size[0]/2,trebrad*np.sin(angle+ 2*np.pi/3) +visual_size[1]/2]).astype(np.int)

        bmline = (    bp[0]-yy+ ((bp[0]-mp[0])/(bp[1]-mp[1]))*(xx-bp[1])   )
        btline = (    bp[0]-yy+ ((bp[0]-tp[0])/(bp[1]-tp[1]))*(xx-bp[1])   )
        mtline = (    mp[0]-yy+ ((mp[0]-tp[0])/(mp[1]-tp[1]))*(xx-bp[1])   )

        pcolorax.imshow(np.logical_and(btline>0,np.logical_and(mtline<0,bmline<0)),cmap=colormap)
        pcolorax.invert_yaxis()
        img = fig2data_trans(pcolorfig)[:,:,0:3]
        plt.cla()

    writing = writing_objects[0]
    if writing == 0:
        markovtext(vis_objects,writing_objects)    
        
    return img

#%%
#picture glitch
#can actually be added onto any module if wanted.
def picglitch(vis_objects,sound_objects,pic,invert):

#    t1 = time.time()
    maxgridvalue = sound_objects[3]
    stream=sound_objects[0]
    CHUNK=sound_objects[1]
    glitch_fact = vis_objects[9]
    visual_size = vis_objects[1]
    volume = sound_objects[4]

    
    picchange = np.zeros((np.shape(pic)[0],np.shape(pic)[1],np.shape(pic)[2]))+255
    if np.random.rand()>0.7:
        picchange = picchange-255
       
    data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
    T = 1.0 / (12*CHUNK)
    yf = np.abs(np.fft.fft(data))*volume
    yf = 2.0/CHUNK *np.abs(yf[:CHUNK//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), int(CHUNK/2))
    
    bass_ind = next(x[0] for x in enumerate(xf) if x[1] > 250) #Hz hopefully
#    mid_ind = next(x[0] for x in enumerate(xf) if x[1] > 2000)
    
    bass_vol = np.max(yf[0:bass_ind])*glitch_fact
    mid_vol = np.max(yf[bass_ind:-1])*glitch_fact
    bass_vol = np.int(((bass_vol/maxgridvalue))*1)
    mid_vol = np.int(((mid_vol/maxgridvalue))*25)+100
    bass_vol = bass_vol*((-1)**np.random.randint(2))
    mid_vol = mid_vol*((-1)**np.random.randint(2))
    
    if bass_vol>np.shape(pic)[0] or mid_vol>np.shape(pic)[0]:
        bass_vol=np.random.randint(visual_size[0])
        mid_vol=np.random.randint(visual_size[0])

       
    #generate interesting-toned separate image
    random_col = np.random.randint(3)
    picchange[:,:,random_col] = pic[:,:,random_col]
    if np.random.rand() > 0.75:
        random_col2 = np.random.randint(3)
        picchange[:,:,random_col2]= pic[:,:,random_col2]
    picchange = picchange.astype(int)
    
    pic = np.concatenate((pic[:,bass_vol:,:],pic[:,:bass_vol,:]),axis=1)
    picchange = np.concatenate((picchange[:,mid_vol:,:],picchange[:,:mid_vol,:]),axis=1)
    
    weight = 0.5
    finalpic = np.copy(pic)
    finalpic[:,:,random_col] = finalpic[:,:,random_col]*weight + picchange[:,:,random_col]*(1-weight)
    finalpic = finalpic/255.
    
    
#    print(mid_vol)
    cutoff = 105
#    print(mid_vol)
    if invert == 1:
        if np.abs(mid_vol)>cutoff+14:
            finalpic = 1-finalpic
    if np.abs(mid_vol)<cutoff:
        img = pic/255.
    elif np.abs(mid_vol)>=cutoff:
        img = finalpic
           
    return img
    
    
#%%
#text on top of image. not really a module, but an effect added on top  
        
def markovtext(vis_objects,writing_objects):

    starttime = writing_objects[2]
    sentence = writing_objects[3]
    timelength = writing_objects[4]
    loc = writing_objects[5]
#    randmod = writing_objects[6]
#    mod_chance = writing_objects[7]
    double = writing_objects[8]

    visual_size = vis_objects[1]
#    pcolorfig = vis_objects[4]
#    pcolorax = vis_objects[5]
    xloc,yloc = loc[0],loc[1]
    xloc = xloc*visual_size[0]
    yloc = yloc*visual_size[1]
    
    wordlist = sentence.split()
    wordtime = timelength/len(wordlist)
    timelist = starttime + np.arange(0,timelength,wordtime)  

#    pcolorax.imshow(np.zeros((visual_size[1],visual_size[0])),cmap='binary')
#    pcolorax.axis('off')

    for i in range(len(wordlist)):
        if time.time() > timelist[i]:
            if double == 0:
                plt.text(xloc[i],yloc[i],wordlist[i], transform = None, fontsize=25,fontname='Helvetica',
                         color='w').set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            if double == 1:
                plt.text(xloc[i],yloc[i],wordlist[i]+'                        '+wordlist[i], transform = None, fontsize=25,fontname='Helvetica',
                         color='w').set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
         
    #maybe one day I can figure out what to do with the picglitch module and text...
    #the following technically works but slows down performance to a crawl.
#    if mod_chance[2] < randmod <= mod_chance[3]: #what to do in picglitch time. could happen elsewhere...
#        ax = vis_objects[14]
#        txt = np.zeros(len(wordlist))
##        fig = vis_objects[13]
#        for i in range(len(wordlist)):
#            if time.time() > timelist[i]:
#                ax.text(xloc[i],yloc[i],wordlist[i], transform = None, fontsize=25,fontname='Helvetica',
#                         color='w').set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
#%%
#tiling pictures/videos
def vidtile(vis_objects,sound_objects,video,writing_objects,copynum,sizes,xlocs,ylocs):
    
#    stream=sound_objects[0]
#    CHUNK=sound_objects[1]
#    volume = sound_objects[4]
    starttime = writing_objects[2]
    vids = vis_objects[7]  
    timelength = writing_objects[4]

    #video frame stuff
    t = time.time()-starttime
    framet = t%vids[video] #loop if video is too short
    vid_name = os.path.splitext(os.listdir(dir_path+'\\videos')[video])[0]
    nframes=len(os.listdir(dir_path+'\\vid_pics\\'+vid_name))
    framenum = int((framet/vids[video])*nframes)

    framepic = io.imread(dir_path+'\\vid_pics\\'+vid_name+'\\'+str(framenum)+'.png')
    framepic=framepic.astype(np.float32)/(255.)
    
    timelist = starttime + np.arange(0,timelength,timelength/copynum)  
    
    bigframe = np.zeros((np.shape(framepic)[0]*3,np.shape(framepic)[1]*3,np.shape(framepic)[2]))
    bigframe[np.shape(framepic)[0]:2*np.shape(framepic)[0],np.shape(framepic)[1]:2*np.shape(framepic)[1],:] = framepic

    for i in range(copynum):
        if time.time() > timelist[i]:
#            smallpic = rescale(framepic,1/sizes[i]) #very slow after 1 copy... could use as a glitch effect tho?
            smallpic = np.copy(framepic)
            sxlen = np.shape(smallpic)[0]
            sylen = np.shape(smallpic)[1]
            x1 = ylocs[i]+np.shape(framepic)[0]
            y1 = xlocs[i]+np.shape(framepic)[1]
            bigframe[x1:x1+sxlen, y1:y1+sylen, :] = smallpic
            
    finalpic = bigframe[np.shape(framepic)[0]:2*np.shape(framepic)[0],np.shape(framepic)[1]:2*np.shape(framepic)[1],:]
    return finalpic
#%%
#notegrid_explode
        
#%%
#something with two sets of random numbers multiplied by each other
        
#%%
#basic frequency amplitude with cool colormap stuff
            
#%%    
#oscilloscope fun
    
#%%
#segment video rows- glitch overlay

#%%
#kaleidescope
    
#%%
#wrap song data into picture. np.reshape
        
        
        
        
        
        