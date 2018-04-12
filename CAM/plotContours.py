import numpy as np
import objectDefinitions as oD
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.widgets import Slider, Button, RadioButtons

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
	fig = plt.figure(1, figsize=(5,5), dpi=90)
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

	def plotContours(self):

		self.slices = []

		for slicedContour in self.contourSlices:
			combinedPoly = combinePolygons(slicedContour)
			vertices_X = np.array([])
			vertices_Y = np.array([])
			if type(slicedContour[0]) is not Polygon:
				print(type(slicedContour[0]))
				for contours in slicedContour:
					vertices = np.array([x.Coordinates[0:2] for x in contours])
					vertices_X = np.concatenate((vertices_X , vertices[:, 0]))
					vertices_Y = np.concatenate((vertices_Y , vertices[:, 1]))
			else:
				vertices_X, vertices_Y = getPolyXY(combinePolygons(slicedContour))

			self.slices.append([vertices_X, vertices_Y])

		self.fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.25, bottom=0.25)

		self.l, = plt.plot(self.slices[0][0], self.slices[0][1], lw=2, color='red')

		plt.axis([np.min(vertices_X)-5, np.max(vertices_X)+5, np.min(vertices_Y)-5, np.max(vertices_Y)+5])

		axcolor = 'lightgoldenrodyellow'

		layerNum = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

		self.layer = Slider(layerNum, 'Layer Number', 0, len(self.slices)-1, valinit=0, valstep=1)

		self.layer.on_changed(self.update)


		plt.show()


	def update(self, val):
		print(val)
		self.l.set_xdata(self.slices[int(val)][0])
		self.l.set_ydata(self.slices[int(val)][1])
		self.fig.canvas.draw_idle()


