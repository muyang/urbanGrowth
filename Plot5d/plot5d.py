import pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
#import seaborn
import pandas as pd
from matplotlib import colors


fig, ax = plt.subplots(1, 1)
ax = fig.add_subplot(111, projection = '3d')

#data= np.loadtxt('1000M_thined.obj')*-1
#data= X =np.random.random((50,6))
data=pd.read_table("./data.txt")
scale=150 #size objective multiplier  

#Plotting 5 objectives:
colors_list = ['red', 'black', 'blue']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2]
norm = colors.BoundaryNorm(bounds, cmap.N)
x=data["a"]
y=data["b"]
z=data["Figure_of_merit"]
c=data["nb"]
s=data["r"]
im= ax.scatter(x, y, z, c=c, s=s, alpha=0.5, cmap=cmap, picker=True)

ax.plot(x, y, 'k+', zdir='z', markersize=0.5)
ax.plot(x, z, 'r+', zdir='y', markersize=0.5)
ax.plot(y, z, 'g+', zdir='x', markersize=0.5)

#Setting the  main axis labels:
# ax.set_xlim(0,3)
# ax.set_ylim(0,1)
ax.set_zlim(0.22,0.265)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('Figure of merit')

#Setting colorbar and its label vertically:
cbar= fig.colorbar(im)
cbar.ax.set_ylabel('nb')
		
#Setting size legend:
objs=s
max_size=6	#np.amax(objs)*scale  #/32.0
min_size=1	#np.amin(objs)*scale  #*-200
handles, labels = ax.get_legend_handles_labels()
display = (0,1,2)

size_max = plt.Line2D((0,1),(0,0), color='k', markersize=max_size, linestyle='')
size_min = plt.Line2D((0,1),(0,0), color='k', markersize=min_size, linestyle='')
'''
legend1=ax.legend(
            [handle for i,handle in enumerate(handles) if i in display]+[size_max,size_min],
            [label for i,label in enumerate(labels) if i in display]+["%.2f"%(np.amax(objs)), "%.2f"%(np.amin(objs))], 
			labelspacing=1.5, 
			title='r', 
			loc=1, 
			frameon=True, 
			numpoints=1, 
			markerscale=1)
'''
legend1= ax.legend([handle for i,handle in enumerate(handles) if i in display]+[size_max,size_min],
        [label for i,label in enumerate(labels) if i in display]+["%.2f"%(np.amax(objs)), "%.2f"%(np.amin(objs))], labelspacing=1.5, title='r', loc=1, frameon=True, numpoints=1, markerscale=1)

#Setting the picker function:

def onpick(event):
   ind = event.ind
   print ('index: %d\nobjective 1: %0.2f\nobjective 2: %0.2f\nobjective 3: %0.2f\nobjective 4: %0.2f\nobjective 5: %0.2f\nobjective 6: %0.2f' % (event.ind[0],data[ind,0],data[ind,1],data[ind,2],data[ind,3],data[ind,4],data[ind,5]))
fig.canvas.mpl_connect('pick_event', onpick)

plt.show()