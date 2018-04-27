import numpy as np
import objectDefinitions as oD
import matplotlib.pyplot as pyplot
from mpl_toolkits import mplot3d
from shapely.geometry import Polygon
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

def combinePolygons( polygonList ):
	flattenedList = []
	for polygon in polygonList:
		if (type(polygon) == list) or polygon.geom_type == "MultiPolygon" :
			for subPolygon in polygon:
				flattenedList.append(subPolygon)
		else:
			flattenedList.append(polygon)
	return flattenedList

def drawPoly ( polygon , ax ):
	"""
	Draws a polygon to ax.
	"""
	fig = pyplot.figure(1, figsize=(5,5), dpi=90)
	if ax == []:
		ax = fig.add_subplot(111)

	x, y = getPolyXY(polygon)
	ax.plot(x, y, color='#6699cc', alpha=0.7,
				linewidth=3, solid_capstyle='round', zorder=2)
	return ax

def getPolyXY ( polygon ):
	if hasattr(polygon, "geom_type"):
		if (polygon.geom_type != "MultiPolygon") or (type(polygon) == list) :
			polygon = [polygon]
	
	x = []
	y = []
	for poly in polygon:
		x_, y_ = poly.exterior.xy
		x += x_
		y += y_
		for insidePolygon in poly.interiors:
			x_, y_ = insidePolygon.xy
			x += x_
			y += y_

	return x, y

class ContourPlot(object):
	"""docstring for ContourPlot"""
	def __init__(self, contourSlices):
		super(ContourPlot, self).__init__()
		self.contourSlices = contourSlices

	def plot(self):
		
		self.slices = []

		for slicedContour in self.contourSlices:
			if not slicedContour:
				continue
			vertices_X = np.array([])
			vertices_Y = np.array([])
			if isinstance(slicedContour, list) and type(slicedContour[0]) is not Polygon:
				for contours in slicedContour:
					vertices = np.array([x.Coordinates[0:2] for x in contours])
					vertices_X = np.concatenate((vertices_X , vertices[:, 0]))
					vertices_Y = np.concatenate((vertices_Y , vertices[:, 1]))
			elif isinstance(slicedContour, list):

				vertices_X, vertices_Y = getPolyXY(combinePolygons(slicedContour))
			else:
				vertices_X, vertices_Y = getPolyXY(slicedContour)

			self.slices.append([vertices_X, vertices_Y])

		self.fig, ax = pyplot.subplots()
		pyplot.subplots_adjust(left=0.25, bottom=0.25)
		ax.set_aspect('equal')
		self.l, = pyplot.plot(self.slices[0][0], self.slices[0][1], lw=2, color='red')

		pyplot.axis([np.min(vertices_X)-5, np.max(vertices_X)+5, np.min(vertices_Y)-5, np.max(vertices_Y)+5])
		# pyplot.axis([-30, 70, -70, 30])
		axcolor = 'lightgoldenrodyellow'

		layerNum = pyplot.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

		self.layer = Slider(layerNum, 'Layer Number', 0, len(self.slices)-1, valinit=0, valstep=1)

		self.layer.on_changed(self.update)


		# pyplot.show()


	def update(self, val):
		self.l.set_xdata(self.slices[int(val)][0])
		self.l.set_ydata(self.slices[int(val)][1])
		self.fig.canvas.draw_idle()



class toolPathSlicePlot(object):
	"""docstring for ContourPlot"""
	def __init__(self, toolpathSlices):
		super(toolPathSlicePlot, self).__init__()
		self.toolpathSlices = toolpathSlices

	def plot(self):
		
		vertices_X = self.toolpathSlices[0]
		vertices_Y = self.toolpathSlices[1]
		vertices_Z = self.toolpathSlices[2]

		self.fig = pyplot.figure('Tool Paths')
		ax = self.fig.gca(projection='3d')
		axcolor = 'lightgoldenrodyellow'

		pyplot.subplots_adjust(left=0.25, bottom=0.25)
		self.l, = pyplot.plot(vertices_X[0], vertices_Y[0], vertices_Z[0], lw=2, color='red')
		if self.meshInfo:
			ax.set_xlim(self.meshInfo.meanX - self.meshInfo.max_range, self.meshInfo.meanX + self.meshInfo.max_range)
			ax.set_ylim(self.meshInfo.meanY - self.meshInfo.max_range, self.meshInfo.meanY + self.meshInfo.max_range)
			ax.set_zlim(self.meshInfo.meanZ - self.meshInfo.max_range, self.meshInfo.meanZ + self.meshInfo.max_range)
			self.modelPlot = mplot3d.art3d.Poly3DCollection(self.meshInfo.meshData.vectors)
			ax.add_collection3d(self.modelPlot)
			rax = pyplot.axes([0.7, 0.05, .2, 0.04], facecolor=axcolor)
			
			self.isVisible = True
			self.button = Button(rax, 'Model Visibility')
			self.button.on_clicked(self.buttonCallback)

		else:

			pyplot.axis([np.min(vertices_X[0])-5, np.max(vertices_X[0])+5, np.min(vertices_Y[0])-5, np.max(vertices_Y[0])+5])

		zFlat = np.concatenate(vertices_Z).ravel()

		# pyplot.axis([-30, 70, -70, 30])

		layerNum = pyplot.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

		self.layer = Slider(layerNum, 'Layer Number', 0, len(vertices_X)-1, valinit=0, valstep=1)

		self.layer.on_changed(self.update)

		# pyplot.show()

	def buttonCallback(self,event):
		self.isVisible = not self.isVisible
		self.modelPlot.set_visible(self.isVisible)

		self.fig.canvas.draw_idle()

	def update(self, val):
		self.l.set_xdata(self.toolpathSlices[0][int(val)])
		self.l.set_ydata(self.toolpathSlices[1][int(val)])
		self.l.set_3d_properties(zs=self.toolpathSlices[2][int(val)])
		self.fig.canvas.draw_idle()

class toolPathCometPlot(object):
	"""docstring for ContourPlot"""
	def __init__(self, toolPaths):
		super(toolPathCometPlot, self).__init__()
		self.toolPaths = toolPaths
		self.vertices_X = toolPaths[0]
		self.vertices_Y = toolPaths[1]
		self.vertices_Z = toolPaths[2]
		self.startVal = 100
		self.meshInfo = None

	def plot(self):

		self.fig = pyplot.figure()
		ax = self.fig.gca(projection='3d')

		pyplot.subplots_adjust(left=0.25, bottom=0.25)

		self.lBold, = pyplot.plot(self.vertices_X[0:10], self.vertices_Y[0:10],
							  self.vertices_Z[0:10], lw=2, color='blue')

		self.lThin, = pyplot.plot(self.vertices_X[10:self.startVal], self.vertices_Y[10:self.startVal],
							  self.vertices_Z[10:self.startVal], lw=.25, color='red')

		if self.meshInfo:

			ax.set_xlim(self.meshInfo.meanX - self.meshInfo.max_range, self.meshInfo.meanX + self.meshInfo.max_range)
			ax.set_ylim(self.meshInfo.meanY - self.meshInfo.max_range, self.meshInfo.meanY + self.meshInfo.max_range)
			ax.set_zlim(self.meshInfo.meanZ - self.meshInfo.max_range, self.meshInfo.meanZ + self.meshInfo.max_range)
		else:
			pyplot.axis([np.min(self.vertices_X)-5, np.max(self.vertices_X)+5, 
						 np.min(self.vertices_Y)-5, np.max(self.vertices_Y)+5])
		axcolor = 'lightgoldenrodyellow'

		moveNum = pyplot.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

		self.move = Slider(moveNum, 'Move Number', 0, len(self.vertices_X)-1, valinit=0, valstep=1)

		self.move.on_changed(self.update)
		# pyplot.show()

	def update(self, val):
		if val > 200:
			self.startVal =  int(val - 200)
		
		self.lBold.set_xdata(self.vertices_X[self.startVal:int(val)])
		self.lBold.set_ydata(self.vertices_Y[self.startVal:int(val)])
		self.lBold.set_3d_properties(zs=self.vertices_Z[self.startVal:int(val)])

		self.lThin.set_xdata(self.vertices_X[0:self.startVal])
		self.lThin.set_ydata(self.vertices_Y[0:self.startVal])
		self.lThin.set_3d_properties(zs=self.vertices_Z[0:self.startVal])
		
		self.fig.canvas.draw_idle()