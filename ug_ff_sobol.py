#based on ug10.py

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  #to plot or visualize the results, you can install matplotlib with 'pip install matplotlib'
from matplotlib import colors
import pylab as PL

from datetime import datetime
#from numba import jit
import os
import sys

from skimage.util import view_as_windows as viewW
from skimage.io import imread
#from skimage.viewer import ImageViewer

#import scipy.ndimage as ndimage
###################
def init():
	np.random.seed()
	global X, width, height, P, S_ij, args, a, b, radius, size, directory, Dmn_copy, nb_type, nb_type_index, id #,Dmn
	global X16_obs, X16_sim, X86

	args = sys.argv  #return arguments list to args
	a,b,radius,nb_type_index,id = float(args[1]),float(args[2]),int(args[3]), int(args[4]), int(args[5])   #args[0] is the name of py script, e.g. 'ug10.py'
	#print nb_type_index
	nb_type=getNeighborType(nb_type_index)
	#print nb_type
	#0,1,2 =moore,vonNeum,vonNeumCircle
	size = 2*radius+1
	
	#X=ndimage.imread('data2/86Stacked_class41.tif')
	X=imread('./data2/lu86_final.tif')
	X[X==256]=0
	X[X==9]=0
	
	# Initializing nx,ny
	width = X.shape[1]
	height = X.shape[0]

	P = np.zeros([height, width])
	S_ij=imread('./data2/su86_final.tif')     #/15*np.random.random((height,width))

	Dmn_copy = Dmn_copy()
	'''
	directory='./outputs2/' + str(a) + '_' + str(b) + '_' + str(radius)
	if not os.path.exists(directory):
		os.makedirs(directory)
	'''
	directory='./outputs/' #do not create folders,just results .txt under the path "./output"
	
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
	Imn_copy = patches(buildup_or_not,[size,size]) * nb_type
	N_copy = np.exp(-b*Dmn_copy)*Imn_copy	
	return np.sum(np.sum(N_copy,axis=2),axis=2)
'''
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
'''

def drawRes():
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

####### new for final3 version ######
'''
def test_func(window):
	Imn = np.array(window).reshape((size,size))
	arr = np.exp(-b*Dmn)*Imn
	arr[radius,radius]=0
	return arr.sum()                    #window=Imn
'''

def footprint_moore():
	arr=np.ones((2*radius+1,2*radius+1))
	#arr[radius,radius]=0
	return arr

def footprint_vonNeumann():
	arr=np.zeros((2*radius+1,2*radius+1))
	arr[radius,:]=1
	arr[:,radius]=1
	return arr
	
def footprint_vonNeumannCircle():
	arr=np.zeros((2*radius+1,2*radius+1))
	y,x = np.ogrid[-radius:radius+1, -radius:radius+1]   #-3,3
	mask = x*x + y*y <= radius*radius
	arr[mask] = 1
	return arr	

def getNeighborType(index):
	if index==0:
		return footprint_moore()
	elif index==1:
		return footprint_vonNeumann()
	elif index==2:
		return footprint_vonNeumannCircle()
	else:
		print "Error: neightborhood type error, only could be 0,1 or 2"

'''	
a, b = 1, 1
r = 3
n=2*r+1
y,x = np.ogrid[-r:r+1, -r:r+1]   #-3,3
mask = x*x + y*y <= r*r
array = np.ones((n, n))
array[mask] = 255
'''
##################################### 	
	
def main():
	print('start time:' + str(datetime.now()))           #466 us
	global nb_type
	init()   #1.8, 0.3, 2, 2
	global buildup_or_not, T
	T=0
	while T <= 100:
		T+=1	
		buildup_or_not = ((X==1)+(X==9)).astype(int)     #10.3ms   ##buildup: 1 #  '+' equals to 'or', and '*' is 'and'
		N_ij=N()  #ndimage.generic_filter(buildup_or_not,test_func,footprint=footprint)   # 25.5s
		#N_ij=ndimage.generic_filter(buildup_or_not,test_func,footprint=footprint_vonNeumann)   # 25.5s 		
		V_ij=1.0 + (-np.log(np.random.rand(height,width)))**a
		Cons=((X!=1)*(X!=9)*(X!=3)*(X!=5)*(X!=0)).astype(int)
		P=Cons*S_ij*N_ij*V_ij
		#except lu=1 or 9
		P2=np.sort(P,axis=None)[::-1] # reversed sort, from max to min
		threshold=P2[329]  # taking the 463rd value of Probability as the threshold #except buildup area (lu==4 or 9)
		#print('Threshold: ',threshold)
		#np.place(X,(X!=4) * (X!=9) * (P>=threshold), 9)  #9==new_urban
		np.place(X,P>=threshold, 9)  #9==new_urban
		#np.place(buildup_or_not,P>=threshold, 1)
		
		#num_of_buildup = np.count_nonzero(X == 1) + np.count_nonzero(X == 9)
		#urban_area = (num_of_buildup) / 100  # km2
		#print('step ', T, 'urban area is ', urban_area, ' km2', str(datetime.now()))
		#if T % 10 == 0:
		#	drawRes()
		#	np.savetxt(directory + '/LU_' + str(T) + '.out', X, delimiter=',')

		if T==100:
			#np.savetxt(directory + '/LU_' + str(a) + '_' + str(b) + '_' + str(radius) + '_' + str(nb_type_index) + '.out', X, delimiter=',')

			X86=imread('./data2/lu86_final.tif')
			X16_obs=imread('./data2/lu16_final.tif')  #obs

			np.place(X86,(X86==9),0)
			np.place(X16_obs,(X16_obs==9),0)
			np.place(X86,(X86!=1),0)
			np.place(X16_obs,(X16_obs!=1),0)	
			np.place(X16_obs,(X16_obs-X86)==1,9)     #0 to 1: new urban area, observed

			#X16_sim=X
			np.place(X,(X!=1)*(X!=9),0)     #9 is new urban area
			A=len(np.where((X16_obs==9)*(X==0))[0])   #(0,1,0) A:-1
			B=len(np.where((X16_obs==9)*(X==9))[0]) #- 10462 - 7741376   #(0,0,9) A:9
			D=len(np.where((X16_obs==0)*(X==9))[0])    #(0,1,8) B:8
			merit_fig=float(B) / (A+B+D)
			#print A,B,D, merit_fig
			#np.savez_compressed(directory + '/LU_' + str(a) + '_' + str(b) + '_' + str(radius) + '_' + str(nb_type_index) + '.out', X, delimiter=',')	
			#np.savetxt(directory + '/merit_fig_' + str(a) + '_' + str(b) + '_' + str(radius) + '_' + str(nb_type_index) + '.txt', np.array(merit_fig), delimiter=' ')	
			
			file = open(directory + str(id) + '_merit_fig_' + str(a) + '_' + str(b) + '_' + str(radius) + '_' + str(nb_type_index) + '.txt', 'w')
			file.write(str(a) + ' ' + str(b) + ' ' + str(radius) + ' ' + str(nb_type_index) + ' ' + str(merit_fig))
			file.close()			


	print('end time:' + str(datetime.now()))
	
if __name__=="__main__":
	main()	