import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib import colors

def visualize3DData (X,scale,cmap):
    """Visualize data in 3d plot with popover next to mouse position.
    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
    Returns:
        None
    """
    fig = plt.figure(figsize = (16,10))
    ax = fig.add_subplot(111, projection = '3d')
    colors_list = ['lightsalmon', 'greenyellow', 'royalblue']
    cmap = colors.ListedColormap(colors_list)
    im= ax.scatter(X[:, 0], X[:, 1], X[:, 4], c= X[:, 3], s= X[:, 2]*scale, cmap=cmap, alpha=0.3, picker = True)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('merit')
    ax.set_zlim(0.22,0.265)
    ax.plot(X[:, 0], X[:, 1], 'k+', zdir='X[:, 4]', markersize=0.5)
    ax.plot(X[:, 0], X[:, 4], 'r+', zdir='X[:, 1]', markersize=0.5)
    ax.plot(X[:, 1], X[:, 4], 'g+', zdir='X[:, 0]', markersize=0.5)

    cbar= fig.colorbar(im)
    loc=np.arange(3)+0.5
    cbar.set_ticks(loc)
    cbar.set_ticklabels(['Moore', 'Neumann', 'NeumannCircle'])  #0,1,2	
    cbar.ax.set_ylabel('nb')

    objs=X[:,2] #r
    
    max_size=np.amax(objs)#*scale/32.0*10
    min_size=np.amin(objs)#*scale/4.5*10
    handles, labels = ax.get_legend_handles_labels()
    display = (0,1,2)

    size_max = plt.Line2D((0,1),(0,0), color='k', marker='o', markersize=max_size,linestyle='')
    size_min = plt.Line2D((0,1),(0,0), color='k', marker='o', markersize=min_size,linestyle='')
    legend1= ax.legend([handle for i,handle in enumerate(handles) if i in display]+[size_max,size_min],
           [label for i,label in enumerate(labels) if i in display]+["%.2f"%(np.amax(objs)), "%.2f"%(np.amin(objs))], labelspacing=1.5, title='r', loc=1, frameon=True, numpoints=1, markerscale=1)


    def distance(point, event):
        """Return distance between mouse position and given data point
        Args:
            point (np.array): np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)


    def annotatePlot(X, index):
        """Create popover label in 3d chart
        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2,_ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
        annotatePlot.label = plt.annotate("index: %d" % index,
            xy = (x2, y2), xytext = (-20,20), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        fig.canvas.draw()


    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        closestIndex = calcClosestDatapoint(X, event)
        annotatePlot (X, closestIndex)

    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
    plt.show()


if __name__ == '__main__':
    #import seaborn
    #X=np.loadtxt('1000M_thined.obj')*-1
    X=np.loadtxt("./data.txt", delimiter='\t', skiprows=1)
    #X = np.random.random((50,6))
    scale=10
    cmap=plt.cm.spectral
    visualize3DData (X,scale,cmap)
