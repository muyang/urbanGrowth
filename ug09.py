import numpy as np
#import scipy.ndimage as ndimage
#from scipy.misc import imsave
#from scipy import misc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  #to plot or visualize the results, you can install matplotlib with 'pip install matplotlib'
from matplotlib import colors
import pylab as PL

from datetime import datetime
#from numba import jit
import os

from skimage.util import view_as_windows as viewW
from skimage.io import imread
#from skimage.viewer import ImageViewer

###################
def init(arg1, arg2, arg3):
	np.random.seed()
	global X, width, height, P, S_ij, a, b, radius, size, directory,Dmn_copy #,Dmn
	a,b,radius = arg1, arg2, arg3
	size = 2*radius+1
	#X=ndimage.imread('data2/LU_86.tif')
	X=imread('data2/LU_86_2.tif')
	X[X==256]=0
	X[X==9]=0
	# Initializing nx,ny
	width = X.shape[1]
	height = X.shape[0]

	P = np.zeros([height, width])
	S_ij=imread('data2/bau_suit-86-2.tif')/15*np.random.random((height,width))

	Dmn_copy = Dmn_copy()
	
	directory='./outputs/' + str(a) + '_' + str(b) + '_' + str(radius)
	if not os.path.exists(directory):
		os.makedirs(directory)

	#global background, fresh_water, fish_farm, sabkha, builtup, crop, desert, undeveloped, beach, new_urban, others, cmap
	#global background, urban, crop, lake, desert, fish_farm, reclamed, beach, open_place, new_urban, cmap
	global lu_types, cmap, norm
	lu_types = ['background', 'urban', 'crop', 'lake', 'desert', 'fish_farm', 'reclamed', 'beach', 'open place', 'new urban']
	#background, urban, crop, lake, desert, fish_farm, reclamed, beach, open_place, new_urban =  0,1,2,3,4,5,6,7,8,9
	colors_list = ['white', 'grey', 'green', 'blue', 'yellow', 'black','brown', 'orange', 'pink', 'red']
	cmap = colors.ListedColormap(colors_list)
	bounds = [0,1,2,3,4,5,6,7,8,9,10]
	norm = colors.BoundaryNorm(bounds, cmap.N)
	
##############################

def Cons():
	conditions = (X!=1)*(X!=9)*(X!=3)*(X!=5)*(X!=0)   #  backgroun 255 -> 0'+' equals to 'or', and '*' is 'and'
	return conditions.astype(int)
	
'''
#np.sum(np.sum(out,axis=2),axis=2)
patches(buildup_or_not,[size,size])     #Imn_copy  buildup_or_not (1218,687,7,7)	
np.ones((height,width,size,size))	    #window
'''
def patches(a, patch_shape):     
    side_size = patch_shape
    ext_size = (side_size[0]-1)//2, (side_size[1]-1)//2
    img = np.pad(a, ([ext_size[0]],[ext_size[1]]), 'constant', constant_values=(0))
    return viewW(img, patch_shape)
'''
def Imn_copy():
	return patches(buildup_or_not,[size,size])
'''
	
def Dmn_copy():   #Dmn_copy
	dist=[]
	for i in range(size):
		for j in range(size):
			dist.append( ((i-radius)**2 + (j-radius)**2)**0.5 )  #Dmn
	Dmn = np.array(dist).reshape((size,size))  #in init()
	return np.repeat(Dmn[np.newaxis,:,:],height*width,axis=0).reshape(height,width,size,size)

def N():
	Imn_copy = patches(buildup_or_not,[size,size])
	N_copy = np.exp(-b*Dmn_copy)*Imn_copy	
	return np.sum(np.sum(N_copy,axis=2),axis=2)

def drawRes1():
	PL.cla()
	PL.pcolor(X, vmin = 0, vmax = 9, cmap = cmap)
	PL.axis('image')
	PL.title('t = ' + str(T))
	res=directory + '/res_T' + str(T) + '.png'
	PL.savefig(res)

def drawRes2():
	res=directory + '/res_T' + str(T) + '.png'
	#imsave(res, X)
	#misc.toimage(X, high=np.max(X), low=np.min(X)).save(res)   #0: black; 255: white
	misc.toimage(X, high=0, low=10).save(res)

def drawRes3():
	fig = plt.figure(figsize=(5,5))
	ax = fig.add_subplot(111)
	ax.set_xlim(left=0, right=width)
	ax.set_ylim(bottom=height, top=0)
	cax=ax.imshow(X,cmap=cmap,norm=norm)
	#ax.plot(X,cmap=cmap,norm=norm)
	cbar=fig.colorbar(cax,ticks=np.arange(10))
	loc=np.arange(10)+0.5
	cbar.set_ticks(loc)
	cbar.set_ticklabels(lu_types)
	#fig.show()
	res=directory + '/res_T' + str(T) + '.png'
	fig.savefig(res)	
#@jit	
def saveRes():
	np.savetxt(directory + '/test.out', X, delimiter=',')

def main():
	print('start time: ', str(datetime.now()))           #466 us
	init(1.8, 0.4, 6)
	global buildup_or_not, T
	T=0
	while T <= 100:	
		buildup_or_not = ((X==1)+(X==9)).astype(int)     #10.3ms   ##buildup: 1 #  '+' equals to 'or', and '*' is 'and'
		N_ij=N()  #ndimage.generic_filter(buildup_or_not,test_func,footprint=footprint)   # 25.5s 
		V_ij=1.0 + (-np.log(np.random.rand(height,width)))**a
		Cons=((X!=1)*(X!=9)*(X!=3)*(X!=5)*(X!=0)).astype(int)
		P=Cons*S_ij*N_ij*V_ij
		#except lu=1 or 9
		P2=np.sort(P,axis=None)[::-1] # reversed sort, from max to min
		threshold=P2[200]  # taking the 463rd value of Probability as the threshold #except buildup area (lu==4 or 9)
		#print('Threshold: ',threshold)
		#np.place(X,(X!=4) * (X!=9) * (P>=threshold), 9)  #9==new_urban
		np.place(X,P>=threshold, 9)  #9==new_urban
		#np.place(buildup_or_not,P>=threshold, 1)
		num_of_buildup = np.count_nonzero(X == 1) + np.count_nonzero(X == 9)
		urban_area = (num_of_buildup) / 100  # km2
		if T % 10 == 0:
			drawRes3()
			np.savetxt(directory + '/LU_' + str(T) + '.out', X, delimiter=',')	
		T+=1
		print('step ', T, 'urban area is ', urban_area, ' km2', str(datetime.now()))
	print('end time: ', str(datetime.now()))
	
if __name__=="__main__":
	main()
	