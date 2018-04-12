import numpy as np
import plotContours
from objectDefinitions import *
from processingFunctions import *
import bintrees
from stl import mesh
import copy
import time
import pylab as pl
from matplotlib import collections as mc

from mpl_toolkits import mplot3d
from matplotlib import pyplot
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box
from shapely import affinity
from descartes import PolygonPatch

def main():
	# Load the STL files and add the vectors to the plot
	meshData = mesh.Mesh.from_file('shellLower.stl')

	maxHeight = np.max(meshData.vectors[:, :, 2])
	minHeight = np.min(meshData.vectors[:, :, 2])
	meanHeight = np.mean(meshData.vectors[:, :, 2])

	maxLength = np.max(meshData.vectors[:, :, 1])
	minLength = np.min(meshData.vectors[:, :, 1])
	meanLength = np.mean(meshData.vectors[:, :, 1])
	

	maxWidth = np.max(meshData.vectors[:, :, 0])
	minWidth = np.min(meshData.vectors[:, :, 0])
	meanWidth =np.mean(meshData.vectors[:, :, 0])

	#Length, width, and height of the block to machine the part out of.
	blockDimensions = [np.abs(maxLength-minLength) + 1, np.abs(maxWidth-minWidth) + 1, np.abs(maxHeight-minHeight) + 1 ] 

	slicePlanes = genSlicePlanes(blockDimensions[2], 20, CSys([0,0, minHeight-1], [0,0,1], [1,0,0]))

	meshTopology = genTopology(meshData)
	segments = incrementalSlicing( meshTopology, slicePlanes, np.abs(slicePlanes[0].Point[2]-slicePlanes[1].Point[2]))
	contourSlices = []
	for segment in segments:
		if segment:
			contourSlices.append(contourConstruction(segment))

	
	contourPlot = plotContours.ContourPlot(contourSlices)
	contourPlot.plotContours()

	vertexSlices = []
	for slicedContour in contourSlices:
		vertices = []
		for contour in slicedContour:
			vertices.append(Polygon(np.array([x.Coordinates[0:2] for x in contour])))
		vertexSlices.append(vertices)

	polygonOutside = box(minWidth-1, minLength-1, maxWidth+1, maxLength+1, ccw=True)

	polyList = []
	for sliceGen in vertexSlices:
		poly = MultiPolygon([polygonOutside]+sliceGen)
		poly = MultiPolygon(sorted(poly, key=Within, reverse=True))
		poly = getPolyDepth( poly )
		polyList.append(genToolPath( poly, pathStepSize=2))


	vertex = Vertex([3,3,3])
	# print(type(vertex) is Vertex)
	contourPlot = plotContours.ContourPlot(polyList)
	contourPlot.plotContours()
	# ax = drawPoly(poly.buffer(-1), [])
	# ax.set_title('Polygon')
	# pyplot.show()
	# print("Poly Made")
	# genToolPath(poly)

	# targetObject = Object(meshData, meshTopology)

	# loop = genClosedLoop(meshTopology, plane)
	# # print(loop)

	# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(meshData.vectors))

	# Auto scale to the mesh size
	# scale = meshData.points.flatten(-1)
	# axes.auto_scale_xyz(scale, scale, scale)

	# pyplot.show()

if __name__ == '__main__':
	main()
