from osgeo import gdal, osr
from gdalconst import * 
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors


#read raster, land use, factors
#filename = 'data/Landuse/1986/1986Alexandria_class_Majority.tif'
filename = 'data/land0.tif'
dataset = gdal.Open(filename, GA_ReadOnly)
data=dataset.ReadAsArray()
X=np.flipud(data)

# Initializing nx,ny
nx = dataset.RasterXSize
ny = dataset.RasterYSize

#agri, coast, hosp, inds, rail, rod30, scho, urban, util, water = 'data/agri.tif', 'data/coast.tif', 'data/hosp.tif', 'data/inds.tif', 'data/rail.tif', 'data/rod30.tif', 'data/scho.tif', 'data/urban.tif', 'data/util.tif', 'data/water.tif'
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

factors=[]             #np.array([])
for i in factor_files:
    ds=gdal.Open(i, GA_ReadOnly)
    factors.append(np.flipud(ds.ReadAsArray()))

for i,j in zip(factors,Ws):
    i /= np.amax(i)    #normalization
    i *= j             #multiplying weights 

factors=np.array(factors)
np.place(factors,factors<0, 0.0)
S_ij=np.sum(factors,axis=0)
#def S_ij(iy,ix):
#    return S_ij[iy,ix]
 
#print(S_ij)
#print(len(np.where(S_ij>0.2)[0]))
'''
def S_ij(iy,ix):
    for f,w in zip(factors,Ws):
        f /= np.amax(f)    #normalization
        f *= w             #multiplying weights 
    factors=np.array(factors)
    return np.sum(factors,axis=0)
'''

def get_neighbours(size):
    nb=[]
    for i in range(2*size+1):
        for j in range(2*size+1):
            nb.append( (i-size,j-size) )
    nb.remove( (0,0) )   
    return nb

neighbourhood = get_neighbours(1)
#((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
#print (get_neighbours(2))
#def get_urban():
#    return np.where(X==builtup)

a,b = 1.8, 0.3

 
def N_ij(iy,ix):
    neighbourhood = get_neighbours(5)
    Dmn=np.array( [(dy*dy+dx*dx)**0.5 for dy,dx in neighbourhood] )
    Imn=np.array( [(1 if X[iy+dy,ix+dx] == builtup or X[iy+dy,ix+dx] == new_urban else 0) for dy,dx in neighbourhood] )
    return sum( np.exp(-b*Dmn) * Imn )


#background, fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, sea = 1, 2, 3, 4, 5, 6, 7, 8, 9
fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, new_urban, background = 2,3,4,5,6,7,8,10,255 

# Colours for visualization. Note that for the colormap to work, this list and the bounds list must be one larger than the number of different values in the array.
colors_list = ['blue', 'pink', 'grey', 'green', 'yellow', 'brown', 'orange', 'red', 'white']

cmap = colors.ListedColormap(colors_list)
bounds = [2,3,4,5,6,7,8,10,254]
norm = colors.BoundaryNorm(bounds, cmap.N)

T=0
P = np.zeros((ny, nx))
N = np.zeros((ny, nx))

#urban grow
def grow(X):
    """Iterate the urban area according to the Si."""
    # The boundary of the forest is always empty, so only consider cells
    # indexed from 1 to nx-2, 1 to ny-2
    #X1 = np.zeros((ny, nx))
    neighbourhood = get_neighbours(5)
    global T

    #urban_area = get_urban()
    #for iy,ix in zip(range(5,ny-5),range(5,nx-5)):
    for ix in range(5,nx-5):
        for iy in range(5,ny-5):
        #if X[iy,ix] == fresh_water or X[iy,ix] == sabkha:
        #    P[iy,ix] = 0
        #elif X[iy,ix] == builtup or X[iy,ix] == new_urban:
        #    P[iy,ix] = 1
        #else:
        #Dmn=np.array( [(dy*dy+dx*dx)**0.5 for dy,dx in neighbourhood] )
        #Imn=np.array( [(1 if X[iy+dy,ix+dx] == builtup or X[iy+dy,ix+dx] == new_urban else 0) for dy,dx in neighbourhood] )
        #N[iy,ix] = sum( np.exp(-b*Dmn)* Imn )
            P[iy,ix] = S_ij[iy,ix] * N_ij(iy,ix) * (1+(-math.log(np.random.random()))**a)
        #if P[iy,ix]>0:
        #    print('Pij:', P[iy,ix])
            #print('Dmn:', Dmn)
            #print('Imn:', Imn)
            #print('S:',S_ij[iy,ix])
            #print('Nij:', N[iy,ix])         
            #if P[iy,ix]>0.05:
            #    X[iy,ix] = new_urban
    print(np.where(P>0))
    #np.place(X,P>0, new_urban)
    #top_P = np.argsort(-P)
    #indices = np.dstack( np.unravel_index( np.argsort((-P).ravel()), (ny, nx) ) )

    #for iy,ix in zip(top_P):
    #    X[iy,ix] = new_urban
        #if P > 0.60:
        #    print (P_ij)
        #    #X1[iy+dy,ix+dx] = builtup
        #    X[iy,ix] = builtup
        #    break
    n_urban = np.count_nonzero(X == builtup) + np.count_nonzero(X == new_urban)
    T += 1
    print('tick ', T, 'urban cells: ', n_urban)
    return X

'''
# The initial fraction of the forest occupied by trees.
# Probability of new tree growth per empty cell, and of lightning strike.
forest_fraction = 0.2
p, f = 0.05, 0.001
'''

'''
for i in (np.arange(100)+1000):
    for j in (np.arange(100)+500):
        S=S_ij[i,j]
        N=N_ij(i,j)
        V=1.0+(-math.log(np.random.random()))**a
        P[i,j]=S*N*V
        if S>0 and N>0:
            print(S,N,V,P[i,j])
'''

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.set_xlim(left=0, right=nx)
ax.set_ylim(bottom=0, top=ny)
ax.xaxis.tick_bottom()
#ax.set_axis_off()
im = ax.imshow(X, cmap=cmap, norm=norm)#, interpolation='nearest')

# The animation function: called to produce a frame for each generation.
def animate(i):
    im.set_data(animate.X)
    animate.X = grow(animate.X)
# Bind our grid to the identifier X in the animate function's namespace.
animate.X = X

# Interval between frames (ms).
interval = 100
anim = animation.FuncAnimation(fig, animate, interval=interval)
plt.show()

