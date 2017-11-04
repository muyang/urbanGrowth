from osgeo import gdal, osr	   #to read, write and process geospatial data 
from gdalconst import *		   #to read, write and process geospatial data 

import numpy as np
import scipy.ndimage as ndimage

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  #to plot or visualize the results, you can install matplotlib with 'pip install matplotlib'
from matplotlib import colors

import pylab as PL
import random as RD
from datetime import datetime
from numba import jit

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
	global X, width, height, factor_files,Ws,a,b
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
	'data/hosp_clip.tif',    #Recreation .062
	'data/inds_clip.tif', 
	'data/rail_clip.tif',    #Proximity to Transportation
	'data/rod30_clip.tif', 
	'data/scho_clip.tif', 
	'data/urban_clip.tif', 
	'data/util_clip.tif', 
	'data/water_clip.tif']
	'''
	proximity to Urban Areas	.256
	Proximity to Urtilities		.149 r
	Proximity to Schools		.100 r
	proximity to Recreation 	.062
	Proximity to Transportation	.143
	Distance from Industry		.033
	Proximity to Vegetation 	.034
	proximity to Coastline		.069  r
	Land Use					.132
	Distance from  Lakes		.024  r
	'''
	Ws=[
	0.034, 
	0.069, 
	0.062
	0.033, 
	0.143, 
	0.062, 
	0.100, 
	0.256, 
	0.149, 
	0.024]
	
	#background, fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, sea = 1, 2, 3, 4, 5, 6, 7, 8, 9
	global background, fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, new_urban	
	background, fresh_water, sabkha, builtup, crop, desert, undeveloped, beach, new_urban = 0,2,3,4,5,6,7,8,9 

	# Colours for visualization. Note that for the colormap to work, this list and the bounds list must be one larger than the number of	different values in the array.
	colors_list = ['white', 'blue', 'pink', 'grey', 'green', 'yellow', 'brown', 'orange', 'red']
	global time, neighbourhood, S_ij, P, N_ij,Cons, cmap 
	cmap = colors.ListedColormap(colors_list)
	bounds = [0.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
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
	
	#conditions = (X!=4)*(X!=9)*(X!=2)*(X!=3)*(X!=0)   #  backgroun 255 -> 0'+' equals to 'or', and '*' is 'and'
	#Cons = conditions.astype(int)  #init constraines area
	
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

##############################

def Cons():
	conditions = (X!=4)*(X!=9)*(X!=2)*(X!=3)*(X!=0)   #  backgroun 255 -> 0'+' equals to 'or', and '*' is 'and'
	return conditions.astype(int)

def test_func(window):
	Imn = np.array(window).reshape((size,size))
	arr = np.exp(-b*Dmn)*Imn
	arr[radius,radius]=0
	return arr.sum()                    #window=Imn

def footprint():
	arr=np.ones((2*radius+1,2*radius+1))
	#arr[radius,radius]=0
	return arr

@jit
def drawRes():
	#fig = PL.figure()
	PL.cla()
	PL.pcolor(X, vmin = 0, vmax = 10, cmap = cmap)
	PL.axis('image')
	PL.title('t = ' + str(T))
	res='./outputs/res_T'+str(T)+ '.png'
	PL.savefig(res)
@jit	
def saveRes():
	np.savetxt('./outputs/test.out', X, delimiter=',')

def main():
	print('start time: ', str(datetime.now()))           #466 us
	init()
	global buildup_or_not, T
	T=0
	while T <= 100:	
		buildup_or_not = ((X==4)+(X==9)).astype(int)     #10.3ms   ##buildup: 1 #  '+' equals to 'or', and '*' is 'and'
		N_ij=ndimage.generic_filter(buildup_or_not,test_func,footprint=footprint)   # 25.5s 
		V_ij=1.0 + (-np.log(np.random.rand(height,width)))**a
		Cons=((X!=4)*(X!=9)*(X!=2)*(X!=3)*(X!=0)).astype(int)
		P=Cons*S_ij*N_ij*V_ij
		#except lu=4 or 9
		P2=np.sort(P,axis=None)[::-1] # reversed sort, from max to min
		threshold=P2[429]  # 462: taking the 463rd value of Probability as the threshold #except buildup area (lu==4 or 9)
		#print('Threshold: ',threshold)
		#np.place(X,(X!=4) * (X!=9) * (P>=threshold), 9)  #9==new_urban
		np.place(X,P>=threshold, 9)  #9==new_urban
		#np.place(buildup_or_not,P>=threshold, 1)
		num_of_buildup = np.count_nonzero(X == 4) + np.count_nonzero(X == 9)
		urban_area = (num_of_buildup) / 100  # km2
		if T % 10 == 0:
			drawRes()
			np.savetxt('./outputs/LU_' + str(T) + '.out', X, delimiter=',')	
		T+=1
		print('step ', T, 'urban area is ', urban_area, ' km2', str(datetime.now()))
	print('end time: ', str(datetime.now()))
		
if __name__=="__main__":
	main()
	
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

'''
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
'''
