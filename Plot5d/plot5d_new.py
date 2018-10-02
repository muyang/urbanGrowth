import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib import colors


data=pd.read_table("./data.txt")
x=data["a"]
y=data["b"]
z=data["Figure_of_merit"]
c=data["nb"]
s=data["r"]

colors_list = ['red', 'black', 'blue']
cmap = colors.ListedColormap(colors_list)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
cax = ax.scatter(x, y, z, s=s*5, c=c, cmap=cmap)

plt.show()