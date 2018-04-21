import numpy as np
import plot
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
from shapely.ops import cascaded_union
from shapely import affinity
from descartes import PolygonPatch

def main():
	# Load the STL files and add the vectors to the plot
	meshData = mesh.Mesh.from_file('shellLower.stl')
	meshInfo = dataInfo(meshData) #Gets bounds of mesh

	#Layer Height
	zSpacing = 1.1

	#Length, width, and height of the block to machine the part out of.
	blockDimensions = [np.abs(meshInfo.maxY-meshInfo.minY) + 1, np.abs(meshInfo.maxX-meshInfo.minX) + 1, np.abs(meshInfo.maxZ-meshInfo.minZ) + 1 ] 

	slicePlanes = genSlicePlanes(blockDimensions[2], zSpacing, CSys([0,0, meshInfo.minZ-1], [0,0,1], [1,0,0]))

	meshTopology = genTopology(meshData)
	segments = incrementalSlicing( meshTopology, slicePlanes, np.abs(slicePlanes[0].Point[2]-slicePlanes[1].Point[2]))
	contourSlices = []
	for segment in segments:
		if segment:
			contourSlices.append(contourConstruction(segment))

	# contourPlot = plot.ContourPlot(contourSlices)
	# contourPlot.plotContours()

	vertexSlices = []
	for slicedContour in contourSlices:
		vertices = []
		for contour in slicedContour:
			vertices.append(Polygon(np.array([x.Coordinates[0:2] for x in contour])))
		vertexSlices.append(vertices)

	polygonOutside = box(meshInfo.minX-3, meshInfo.minY-3, meshInfo.maxX+3, meshInfo.maxY+3, ccw=True)

	vertexSlices.reverse() #Make sure that the slices are going down

	unionedPolyList = []
	polyArr = []
	for i, sliceGen in enumerate(vertexSlices, 0):
		polyContourList = sorted(sliceGen, key=Within, reverse=True)

		# Generate the unioned polygon list, which is used to prevent overhangs from
		# being machined when there is material above the overhang that the tool will
		# collide with.
		if unionedPolyList:
			unionedPolyList.append(cascaded_union([unionedPolyList[-1], 
				getPolyDepth(MultiPolygon(polyContourList))]))
		else:
			unionedPolyList.append(getPolyDepth(MultiPolygon(polyContourList)))

		if isinstance(unionedPolyList[-1], MultiPolygon):
			unionedPolySublist = [polygon for polygon in unionedPolyList[-1]] 
		else:
			unionedPolySublist = [unionedPolyList[-1]]

		#Generates the sliced polygon with correct 'inside' and 'outside' properties
		poly = polygonOutside.difference(getPolyDepth(unionedPolySublist))
		polyArr.append(poly)
	
	polyList = []
	x = []
	y = []
	z = []

	#Generate toolpaths that account for the tool geometry
	toolOffsets = generateToolOffsets(unionedPolyList, zSpacing)
	for i, poly in enumerate(toolOffsets, 0):
		poly = polygonOutside.difference(poly)
		toolPaths = genToolPath( poly, pathStepSize=2, initial_Offset=0, 
			zHeight=-i*zSpacing+(len(vertexSlices)+4)*zSpacing/2+meshInfo.meanZ, 
			topBounds = meshInfo.maxZ-meshInfo.minZ+2)

		x.append(toolPaths.XYZ[0])
		y.append(toolPaths.XYZ[1])
		z.append(toolPaths.XYZ[2])

		polyList.append(toolPaths.PolyList)


	toolContours = plot.ContourPlot(toolOffsets)
	toolContours.plot()


	toolpathPlot = plot.toolPathSlicePlot([x, y, z])
	toolpathPlot.meshInfo = meshInfo
	toolpathPlot.plot()
	xF = [item for sublist in x for item in sublist]
	yF = [item for sublist in y for item in sublist]
	zF = [item for sublist in z for item in sublist]

	# toolPathAnimatedPlot = plot.toolPathCometPlot([xF, yF, zF])
	# toolPathAnimatedPlot.plot()

	# fig = pyplot.figure()
	# ax = fig.gca(projection='3d')
	# ax.set_title('3D Toolpath')
	# ax.plot(xF, yF, zF)
	# pyplot.show()


	# ax = drawPoly(poly.buffer(-1), [])
	# ax.set_title('Polygon')
	# pyplot.show()
	# print("Poly Made")
	# genToolPath(poly)

	# targetObject = Object(meshData, meshTopology)

	# loop = genClosedLoop(meshTopology, plane)
	# # print(loop)

	# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(meshData.vectors))

	# # Auto scale to the mesh size
	# scale = meshData.points.flatten(-1)
	# axes.auto_scale_xyz(scale, scale, scale)

	# pyplot.show()

if __name__ == '__main__':
	main()
