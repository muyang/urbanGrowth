from osgeo import gdal, osr	   #to read, write and process geospatial data 
from gdalconst import *		   #to read, write and process geospatial data 
import math

import numpy as np
import scipy.ndimage as ndimage

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  #to plot or visualize the results, you can install matplotlib with 'pip install matplotlib'
from matplotlib import colors

import pylab as PL
import random as RD
from datetime import datetime

###################
##
##
###################
def init():
	#read raster, land use, factors
	filename = 'data/land0.tif'
	dataset = gdal.Open(filename, GA_ReadOnly)

	#read image/raster as numpy array in variable X
	data=dataset.ReadAsArray()
	
	RD.seed()
	global X, width, height, factor_files,Ws, a,b
	a,b = 1.8, 0.3
	
	X=np.flipud(data)
	X[X==255]=0
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

	# Colours for visualization. Note that for the colormap to work, this list and the bounds list must be one larger than the number of	different values in the array.
	colors_list = ['blue', 'pink', 'grey', 'green', 'yellow', 'brown', 'orange', 'red', 'white']
	global time, neighbourhood, S_ij, P, N_ij,Cons, cmap 
	cmap = colors.ListedColormap(colors_list)
	bounds = [2,3,4,5,6,7,8,9,255]
	norm = colors.BoundaryNorm(bounds, cmap.N)
	#time = 0
	P = np.zeros([height, width])
	#use for loop to load rasters for different factors
	factors=[]
	for i in factor_files:
		ds=gdal.Open(i, GA_ReadOnly)
		factors.append(np.flipud(ds.ReadAsArray()))

	for i,j in zip(factors,Ws):
		i /= np.amax(i)	   #i=i/np.amax(i)		 normalization, factor_i/max   
		i *= j			   #i=i*j				 multiplying weights 

	factors=np.array(factors)  # convert list to array
	np.place(factors,factors<0, 0.0)  #replace tiny values in Nodata area with 0
	S_ij=np.sum(factors,axis=0)	  # to calculate Suitability
	
	conditions = (X!=2)*(X!=3)*(X!=0)   #  backgroun 255 -> 0'+' equals to 'or', and '*' is 'and'
	Cons = conditions.astype(int)  #init constraines area
	
	#init window or neighbourhood size / radius
	global radius,size,footprint,Dmn
	radius = 3
	size = 2*radius+1
	footprint = footprint()

	dist=[]
	for i in range(size):
		for j in range(size):
			dist.append( ((i-radius)**2 + (j-radius)**2)**0.5 )  #Dmn
	Dmn = np.array(dist).reshape((size,size))  #in init() 
'''
def draw():
	PL.cla()
	PL.pcolor(config, vmin = 0, vmax = 9, cmap = cmap)
	#PL.pcolor(config, vmin = 0, vmax = 255, cmap = PL.cm.binary)
	PL.axis('image')
	PL.title('t = ' + str(time))
'''
##############################
def Cons():
	conditions = (X!=2)*(X!=3)*(X!=0)   #  backgroun 255 -> 0'+' equals to 'or', and '*' is 'and'
	return conditions.astype(int)

def N():
	buildup_or_not = ((X==4)+(X==9)).astype(int)    ##buildup: 1 #  '+' equals to 'or', and '*' is 'and'
	N_ij=ndimage.generic_filter(buildup_or_not,test_func,footprint=footprint())
	return N_ij
	
def S():
	factors=[]
	for i in factor_files:
		ds=gdal.Open(i, GA_ReadOnly)
		factors.append(np.flipud(ds.ReadAsArray()))

	for i,j in zip(factors,Ws):
		i /= np.amax(i)	   #i=i/np.amax(i)		 normalization, factor_i/max   
		i *= j			   #i=i*j				 multiplying weights 

	factors=np.array(factors)  # convert list to array
	np.place(factors,factors<0, 0.0)  #replace tiny values in Nodata area with 0
	S_ij=np.sum(factors,axis=0)
	#return S_ij

def V():
	return 1.0 + (-np.log(np.random.rand(height,width)))**a

def test_func2(window):
	return window.sum() 

def test_func(window):
	Imn = np.array(window).reshape((size,size))
	arr = np.exp(-b*Dmn)*Imn
	arr[radius,radius]=0
	return arr.sum()                    #window=Imn
	
def footprint():
	arr=np.ones((2*radius+1,2*radius+1))
	#arr[radius,radius]=0
	return arr

def drawRes():
    #save the result when Time = t
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlim(left=0, right=width)
    ax.set_ylim(bottom=0, top=height)
    ax.xaxis.tick_bottom()
    ax.plot(buildup_or_not)
    res='res'+str(T)+ '.png'
    fig.savefig(res)
'''	
def step():
	T=0
	while T < 100:
		P= Cons()*S()*N()*V()
		P2=np.sort(P,axis=None)[::-1] # reversed sort, from max to min
		threshold=P2[462]  # 462: taking the 463rd value of Probability as the threshold
		#print('Threshold: ',threshold)
		np.place(X,P>=threshold, 9)  #9==new_urban
		#np.place(buildup_or_not,P>=threshold, 1)
		T+=1
		#drawRes()
		print('step ', T, str(datetime.now()))
'''

def main():
	print('start time: ', str(datetime.now()))           #466 us
	init()
	global buildup_or_not, T
	T=0
	while T < 10:	
		buildup_or_not = ((X==4)+(X==9)).astype(int)     #10.3ms   ##buildup: 1 #  '+' equals to 'or', and '*' is 'and'
		N_ij=ndimage.generic_filter(buildup_or_not,test_func,footprint=footprint)   # 25.5s 
		V_ij=1.0 + (-np.log(np.random.rand(height,width)))**a
		P=Cons*S_ij*N_ij*V_ij

		P2=np.sort(P,axis=None)[::-1] # reversed sort, from max to min
		threshold=P2[462]  # 462: taking the 463rd value of Probability as the threshold
		#print('Threshold: ',threshold)
		np.place(X,P>=threshold, 9)  #9==new_urban
		#np.place(buildup_or_not,P>=threshold, 1)
		T+=1
		#drawRes()
		print('step ', T, str(datetime.now()))
	print('end time: ', str(datetime.now()))
		
if __name__=="__main__":
	main()
	
'''
def N_fb_xy(radius):
	size = 2*radius+1
	arr=np.ones((size,size))
	arr[radius,radius]=0
	footprint=arr

	dist=[]
	for i in range(size):
		for j in range(size):
			dist.append( ((i-radius)**2 + (j-radius)**2)**0.5 )
	dist = np.array(dist).reshape((size,size))
	
	print footprint
	print dist
'''	

'''
def N_ij(iy,ix):
	xy = get_neighbours(5)
	Dmn=np.array( [(dy*dy+dx*dx)**0.5 for dy,dx in neighbourhood] )
	Imn=np.array( [(1 if X[iy+dy,ix+dx] == 4 or X[iy+dy,ix+dx] == 9 else 0) for dy,dx in neighbourhood] )
	return sum( np.exp(-b*Dmn) * Imn )
'''
'''	
#test_func
def test_func2(i):
	print i
	return i.sum()

#fp=footprint, or neighbourhood

footprint=np.array([
             [1,1,1,1,1],
             [1,1,1,1,1],
			 [1,1,0,1,1],
			 [1,1,1,1,1],
			 [1,1,1,1,1],
])


def N():
	buildup_or_not = ((X==4)+(X==9)).astype(int)  
'''	