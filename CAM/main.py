import pickle
import numpy as np
import plot
from objectDefinitions import *
from processingFunctions import *
from vplusConverter import writeVplus
import bintrees
from stl import mesh
import copy
import time
import pylab as pl
from matplotlib import collections as mc
import paramaters as params
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box
from shapely.ops import cascaded_union
from shapely import affinity
from descartes import PolygonPatch

def main( meshFile, outputFile="out.txt" ):
	# Load the STL files and add the vectors to the plot
	meshData = mesh.Mesh.from_file(meshFile)
	meshData.vectors = meshData.vectors + params.ZERO_OFFSET # Move the mesh into position

	meshInfo = dataInfo(meshData) #Gets bounds of mesh

	blockCenter = params.BLOCK_CENTER

	if params.MV_BLOCK:
		blockCenter = blockCenter + params.ZERO_OFFSET

	Block = BLOCK(blockCenter, params.BLOCK_DIMENSIONS)

	#Length, width, and height of the block to machine the part out of.
	blockDimensions = params.BLOCK_DIMENSIONS

	slicePlanes = genSlicePlanes(blockDimensions[2], params.Z_SPACING, CSys([0,0, Block.Z_MIN], [0,0,1], [1,0,0]))


	# Round planes and vertices so they don't intersect
	for plane in slicePlanes:
		plane.Point[2] = np.around(plane.Point[2], decimals=4)+.00005

	meshTopology = genTopology(meshData)
	for i, vertex in enumerate(meshTopology.vertices, 0):
		meshTopology.vertices[i][2] = np.around(vertex[2], decimals=4)

	# Slice mesh and generate contours
	segments = incrementalSlicing( meshTopology, slicePlanes, params.Z_SPACING)
	contourSlices = []
	for segment in segments:
		contourSlices.append(contourConstruction(segment))

	vertexSlices = []
	for slicedContour in contourSlices:
		vertices = []
		for contour in slicedContour:
			vertices.append(Polygon(np.array([x.Coordinates[0:2] for x in contour])))
		vertexSlices.append(vertices)

	polygonOutside = box(Block.X_MIN, Block.Y_MIN, Block.X_MAX, Block.Y_MAX, ccw=True)

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

	toolPathList = []
	#Generate toolpaths that account for the tool geometry
	toolOffsets = generateToolOffsets(unionedPolyList, params.Z_SPACING, bufferRes=params.BUFFER_RES)
	for i, poly in enumerate(toolOffsets, 0):
		poly = polygonOutside.difference(poly)
		toolPaths = genToolPath( poly, pathStepSize=params.PATH_STEP_SIZE, initial_Offset=params.PATH_INITIAL_OFFSET, 
			zHeight=-i*params.Z_SPACING + Block.Z_MAX, 
			topBounds = Block.Z_MAX+params.SAFE_HEIGHT)
		if not toolPaths:
			continue

		toolPathList.append(toolPaths)
		polyList.append(toolPaths.PolyList)

	x = []
	y = []
	z = []
	for i, toolPath in enumerate(toolPathList, 0):
		x.append(toolPath.XYZ[0])
		y.append(toolPath.XYZ[1])
		z.append(toolPath.XYZ[2])

		if i == len(toolPathList) - 1:
			break

		xs = np.array([x[-1][-1], toolPathList[i+1].XYZ[0][0]])
		ys = np.array([y[-1][-1] , toolPathList[i+1].XYZ[1][0]])
		zs = np.array([Block.Z_MAX+params.SAFE_HEIGHT, Block.Z_MAX+params.SAFE_HEIGHT])
		x[-1] = np.concatenate([x[-1], xs])
		y[-1] = np.concatenate([y[-1], ys])
		z[-1] = np.concatenate([z[-1], zs])

	xF = np.array([item for sublist in x for item in sublist]) 
	yF = np.array([item for sublist in y for item in sublist])
	zF = np.array([item for sublist in z for item in sublist])

	file = open(outputFile, 'w')
	pickle.dump([xF, yF, zF], file)
	file.close()

	toolContours = plot.ContourPlot(toolOffsets)
	toolContours.plot()

	toolpathPlot = plot.toolPathSlicePlot([x, y, z])
	toolpathPlot.meshInfo = meshInfo
	toolpathPlot.plot()

	toolPathAnimatedPlot = plot.toolPathCometPlot([xF, yF, zF])
	toolPathAnimatedPlot.meshInfo = meshInfo
	toolPathAnimatedPlot.plot()

	pyplot.show()




if __name__ == '__main__':
	main('test1.stl')
	writeVplus('out.txt','output.pg')
