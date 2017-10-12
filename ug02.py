from osgeo import gdal, osr    #to read, write and process geospatial data 
from gdalconst import *        #to read, write and process geospatial data 
import math

import numpy as np
#import matplotlib.pyplot as plt  #to plot or visualize the results, you can install matplotlib with 'pip install matplotlib'
#from matplotlib import animation
from matplotlib import colors

import matplotlib
matplotlib.use('TkAgg')

import pylab as PL
import random as RD
import scipy as SP

###################
##
##
###################
RD.seed()

#width = nx
#height = ny

a,b = 1.8, 0.3

'''
def S_ij(iy,ix):
    for f,w in zip(factors,Ws):
        f /= np.amax(f)    #normalization
        f *= w             #multiplying weights 
    factors=np.array(factors)
    return np.sum(factors,axis=0)[iy,ix]
'''

#function to find Moore neighbourhood with radius 'size'
def get_neighbours(size):
    nb=[]
    for i in range(2*size+1):
        for j in range(2*size+1):
            nb.append( (i-size,j-size) )
    nb.remove( (0,0) )   
    return nb

#neighbourhood = get_neighbours(1)
#((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
#print (get_neighbours(2))
#def get_urban():
#    return np.where(X==builtup)

 
def N_ij(iy,ix):
    neighbourhood = get_neighbours(5)
    Dmn=np.array( [(dy*dy+dx*dx)**0.5 for dy,dx in neighbourhood] )
    Imn=np.array( [(1 if X[iy+dy,ix+dx] == builtup or X[iy+dy,ix+dx] == new_urban else 0) for dy,dx in neighbourhood] )
    return sum( np.exp(-b*Dmn) * Imn )


def init():
    #read raster, land use, factors
    filename = 'data/land0.tif'
    dataset = gdal.Open(filename, GA_ReadOnly)

    #read image/raster as numpy array in variable X
    data=dataset.ReadAsArray()
    global X, width, height #X, nx, ny
    X=np.flipud(data)
    # Initializing nx,ny
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    #agri, coast, hosp, inds, rail, rod30(streets), scho, urban, util, water = 'data/agri.tif', 'data/coast.tif', 'data/hosp.tif', 'data/inds.tif', 'data/rail.tif', 'data/rod30.tif', 'data/scho.tif', 'data/urban.tif', 'data/util.tif', 'data/water.tif'
#Ws=[0.034, 0.069, hosp, 0.033, 0.143, rod30, 0.100, 0.256, 0.149, 0.024]

    factor_files = [
    'data/agri_clip.tif', 
    'data/coast_clip.tif', 
    'data/hosp_clip.tif', 
    'data/inds_clip.tif', 
    'data/rail_clip.tif', 
    'data/rod30_clip.tif', 
    'data/scho_clip.tif', 
    'data/urban_clip.tif', 
    'data/util_clip.tif', 
    'data/water_clip.tif']

    Ws=[
    0.034, 
    0.069, 
    0.132, 
    0.033, 
    0.143, 
    0.062, 
    0.100, 
    0.256, 
    0.149, 
    0.024]
    
    #background, fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, sea = 1, 2, 3, 4, 5, 6, 7, 8, 9
    global fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, new_urban, background    
    fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, new_urban, background = 2,3,4,5,6,7,8,9,255 

    # Colours for visualization. Note that for the colormap to work, this list and the bounds list must be one larger than the number of    different values in the array.
    colors_list = ['blue', 'pink', 'grey', 'green', 'yellow', 'brown', 'orange', 'red', 'white']
    global time, config, nextConfig, neighbourhood, S_ij, P, N, cmap 
    cmap = colors.ListedColormap(colors_list)
    bounds = [2,3,4,5,6,7,8,9,255]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    time = 0
    P = N = config = SP.zeros([height, width])
    neighbourhood = get_neighbours(5)

    for ix in xrange(580,width-8):
        for iy in xrange(900,height-8):
            config[iy, ix] = X[iy,ix]
            #config = X

    #use for loop to load rasters for different factors
    factors=[]
    for i in factor_files:
        ds=gdal.Open(i, GA_ReadOnly)
        factors.append(np.flipud(ds.ReadAsArray()))

    for i,j in zip(factors,Ws):
        i /= np.amax(i)    #i=i/np.amax(i)       normalization, factor_i/max   
        i *= j             #i=i*j                multiplying weights 

    factors=np.array(factors)  # convert list to array
    np.place(factors,factors<0, 0.0)  #replace tiny values in Nodata area with 0
    S_ij=np.sum(factors,axis=0)   # to calculate Suitability

    #nextConfig = SP.zeros([height, width])
    #print S_ij, neighbourhood, get_neighbours(1)

def draw():
    PL.cla()
    PL.pcolor(config, vmin = 0, vmax = 9, cmap = cmap)
    #PL.pcolor(config, vmin = 0, vmax = 255, cmap = PL.cm.binary)
    PL.axis('image')
    PL.title('t = ' + str(time))

def step():
    global time, config, nextConfig, neighbourhood

    time += 1

    for ix in xrange(width):
        for iy in xrange(height):
            state = config[iy, ix]

            if state == fresh_water or state == sabkha or state == background:
                P[iy,ix] = 0
            else:
                S=S_ij[iy,ix]
                #S=S_ij(iy,ix)
                N=N_ij(iy,ix)
                V=1.0+(-math.log(np.random.random()))**a
                P[iy,ix]=S*N*V
                #print S, N, V, P[iy,ix]
            if P[iy,ix] > 0.5:
                state = builtup

            config[iy, ix] = state
    #print get_neighbours(1)
    #print config
    #config, nextConfig = nextConfig, config
    n_urban = np.count_nonzero(config == builtup)
    print('step ', time, 'urban cells: ', n_urban)

import pycxsimulator
pycxsimulator.GUI().start(func=[init,draw,step])
