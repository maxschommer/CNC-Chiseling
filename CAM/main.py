import numpy as np
import bintrees
from stl import mesh
import copy
import time
import pylab as pl
from matplotlib import collections  as mc
from sortedcontainers import SortedList, SortedSet, SortedDict
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from shapely.geometry import Polygon, Point, LineString
from descartes import PolygonPatch

def calcPlaneTriangleIntersection( plane, triangle ):
	pointSigns = [1 ,1, 1]
	for i, vector in enumerate(triangle):
		if np.dot(np.subtract( vector, plane.Point), plane.Normal) < 0:
			pointSigns[i] = -1
	
	triangleLoc = np.sum(pointSigns)

	if triangleLoc == -1:
		triTip = pointSigns.index(1)

	elif triangleLoc == 1:
		triTip = pointSigns.index(-1)
	elif np.abs(triangleLoc) == 3:
		return False

	intVecs = []
	for i in range(3):
		if i != triTip:
			intVecs.append(np.subtract(triangle[i], triangle[triTip]))

	L1 = Line(triangle[triTip], intVecs[0])
	L2 = Line(triangle[triTip], intVecs[1])

	p1 = calcPlaneLineIntersection( plane, L1 )
	p2 = calcPlaneLineIntersection( plane, L2 )
	return [p1, p2]

# determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.

def tolerantBinarySearchVertexList( vertex, vertexList , toPrint=False):
	"""Performs iterative binary search to find the position of a vertex in a given, sorted, list.
	vertexList -- sorted list of vertices or points
	vertex -- 3d vertex or point (list or array) you are searching for
	"""

	first = 0
	last = len(vertexList) -1
	if (not isinstance(vertex, list)):
		vertex = vertex.tolist()
	if toPrint:
		print(vertexList)
		print("Vertex to Find = %s" % (vertex))
	while first <= last:
		i = (first + last) / 2
		if toPrint:
			print("Current Index: %s, Vertex = %s" % (i, vertexList[i]))	
		if tolerantEquals(vertexList[i], vertex):
				return i
		elif tolerantCompare( vertexList[i] , vertex) == 1:
				last = i - 1
		elif tolerantCompare( vertexList[i], vertex) == -1:
				first = i + 1
		else:
			return None

def tolerantLinearSearch( vertex, vertexList ):
	for i, vertexTest in enumerate(vertexList):
		
		
		if (not isinstance(vertexTest, list)):
			vertexTest = vertexTest.tolist()
		if tolerantEquals(vertexTest, vertex):
			return i

	return None

def tolerantEquals( v1, v2, tolerance=.000000001 ):

	if (np.abs(v1[0]-v2[0]) < tolerance) and (np.abs(v1[1]-v2[1]) < tolerance) and (np.abs(v1[2]-v2[2]) < tolerance):
		return True
	else:
		return False

def tolerantCompare( v1, v2, tolerance=.000000001 ):
	"""Returns True if v1 > v2 within tolerance, and False otherwise"""

	if (np.abs(v1[0]-v2[0]) < tolerance):
		if (np.abs(v1[1]-v2[1]) < tolerance):
			if (np.abs(v1[2]-v2[2]) < tolerance):
				return 0
			else:
				if v1[2] > v2[2]:
					return 1
		else:
			if v1[1] > v2[1]:
				return 1
	else:
		if v1[0] > v2[0]:
			return 1
	return -1

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def calcPlaneLineIntersection( plane, line ):
	sI = np.dot(plane.Normal, np.subtract(plane.Point, line.Point))/np.dot(plane.Normal, line.Direction)
	return line.Point + sI * line.Direction

def pointPlaneDist( plane, point ):
	pass

def projectPointToPlane( plane, planeX, point, normalize=True):
	if normalize:
		plane.Normal = np.linalg.norm(plane.Normal)
		planeX = np.linalg.norm(planeX)
	x = np.dot((point-plane.Point), planeX)
	y = np.dot((point-plane.Point), np.cross(plane.Normal, planeX))
	return [x,y]

def findMinMaxZ(triangle):
	triangle = np.asarray(triangle)
	return [np.min(triangle[:,2], 0), np.max(triangle[:,2], 0)]


def genTopology ( mesh ):

	vertexList = SortedList(key=cmp_to_key(tolerantCompare))
	faceList = SortedList()
	ind = 0


	for triangle in mesh.vectors:
		# print(ind)
		ind += 1
		for vertex in triangle:
			j = tolerantBinarySearchVertexList( vertex, vertexList )
			if (j == None):
				vertexList.add(vertex.tolist())

	ind = 0
	for triangle in mesh.vectors:
		# print(ind)
		ind += 1
		face = []
		for vertex in triangle:
			j = tolerantBinarySearchVertexList( vertex, vertexList )
			if (j != None):
				face.append(j)
			else:
				tolerantBinarySearchVertexList(vertex, vertexList, True)
				raise ValueError("Couldn't find vertex in list")
		faceList.add(face)

	# # Replace this loop with a loop through three sorted lists for the three vertices
	# # of each triangle. Group the triangles 
	# adjacentFaceList = [0]*len(faceList)
	# for j, face in enumerate(faceList):
	# 	print(j)
	# 	adjacentFaces = []
	# 	for i, isAdjacentFace in enumerate(faceList):
	# 		if j == i:
	# 			continue
	# 		if i in adjacentFaces:
	# 			continue
	# 		for vertex in face:
	# 			if (vertex in isAdjacentFace):
	# 				adjacentFaces.append(i)
	# 				break

	# 	adjacentFaceList[j] = adjacentFaces

	return MeshTopology(faceList, vertexList)


def genClosedLoop( MeshTopology,  plane):
	"""generates a closed loop from the intersection of a plane with a mesh"""
	allIntPoints = []
	vertexList = SortedList(key=cmp_to_key(tolerantCompare))
	pairList = SortedList()

	for i in MeshTopology.faces:
		triangle = getTrianglefromIndex(MeshTopology, i)
		intPoints = calcPlaneTriangleIntersection( plane, triangle )
		
		if intPoints:
			# print(intPoints)
			allIntPoints.append(intPoints)
	# tolerantBinarySearchVertexList([-14.88553333, 5.,-21.80109953], allIntPoints )
	for pair in allIntPoints:
		for vertex in pair:
			j = tolerantLinearSearch( vertex, vertexList )
			k = tolerantBinarySearchVertexList( vertex, vertexList )
			if j != k:
				tolerantBinarySearchVertexList(vertex, vertexList, True)
				print("J = %s, K = %s" % (j, k))
				print("Vertex : %s" %vertex)
			if (j == None):
				if (not isinstance(vertex, list)):
					vertex = vertex.tolist()
				vertexList.add(vertex)
	# for vertex in vertexList:
	# 	if (not isinstance(vertex, list)):
	# 		print("LOL")
	# 		return False
	for pair in allIntPoints:
		pairInd = []

		for vertex in pair:
			if (not isinstance(vertex, list)):
				vertex = vertex.tolist()
			j = tolerantLinearSearch( vertex, vertexList )
			if (j != None):
				pairInd.append(j)
			else:
				print(j)
				print("Attempting Linear Search Pair: ")
				print(tolerantLinearSearch(vertex, allIntPoints))
				print(tolerantBinarySearchVertexList(vertex, allIntPoints))
				print(vertex)
				print("Linear Search Pair Complete")
				raise ValueError("Couldn't find vertex in list")

		pairList.add(pairInd)

	print(pairList)
	newList = recurseClosedLoop(pairList[0], pairList)

	finalList = []

	for sublist in newList:
		del sublist[-1]
		for i in sublist:
			finalList.append(i)

	d = getTrianglefromIndex(MeshTopology, finalList)

	return d

def getTrianglefromIndex ( MeshTopology, face ):
	"""Helper function for genClosedLoop. Returns vertices of triangles from a face"""
	triangle = []
	for i in face:
		triangle.append(MeshTopology.vertices[i])
	return triangle

def recurseClosedLoop(first, pairList):
	"""Helper function for genClosedLoop. Iterates over pairList to find closed loop"""
	new = []
	while pairList != []:
		new.append(first)

		if first in pairList:
			pairList.remove(first)
		else:
			print(pairList)
			pairList.remove(list(reversed(first)))

		for twoInts in pairList:
			for point in twoInts:
				if np.array_equal(point, first[1]):
					if np.all(point == twoInts[1]):
						twoInts = list(reversed(twoInts))
						new += recurseClosedLoop(twoInts, pairList)
					else:
						new += recurseClosedLoop(twoInts, pairList)
	return new

"""
Optional paramaters to be added later for toolpath generation.
Units are in Millimeters
"""
def genToolPath( multiPoly, pathStepSize=12, initial_Offset=0 ):

	polygonInside = [(-100, 0), (0, 100), (100, 0), (0, -100)]
	polygonOutside = [(-200, -200), (-200, 200), (200, 200), (200, -200)]
	# print(polygon.contains(Point(0.25, 0.05)))

	polyList = []
	poly = Polygon(polygonOutside, [polygonInside])

	step = initial_Offset
	while (1):

		poly2 = Polygon(poly).buffer(step)
		print(poly2)
		if poly2.is_empty :
			break
		polyList.append(poly2)
		step = step-pathStepSize

	ax = drawPoly(combinePolygons(polyList), [])
	# print(polyBuffer)

	ax.set_title('Polygon')
	pyplot.show()

def combinePolygons( polygonList ):
	flattenedList = []
	for polygon in polygonList:
		if (type(polygon) == list) or polygon.geom_type == "MultiPolygon" :
			print("Multi")
			for subPolygon in polygon:
				flattenedList.append(subPolygon)
		else:
			flattenedList.append(polygon)
			print("Single")
	return flattenedList

def drawPoly ( polygon , ax ):
	"""
	Draws a polygon to ax.
	"""
	fig = pyplot.figure(1, figsize=(5,5), dpi=90)
	if ax == []:
		ax = fig.add_subplot(111)
	if hasattr(polygon, "geom_type"):
		if (polygon.geom_type != "MultiPolygon") or (type(polygon) == list) :
			polygon = [polygon]

	for poly in polygon:
		x, y = poly.exterior.xy
		ax.plot(x, y, color='#6699cc', alpha=0.7,
				linewidth=3, solid_capstyle='round', zorder=2)
		for insidePolygon in poly.interiors:
			x, y = insidePolygon.xy
			ax.plot(x, y, color='#6699cc', alpha=0.7,
				linewidth=3, solid_capstyle='round', zorder=2)
	return ax

def genSlicePlanes( maxHeight, numPlanes, originCSys ):
	"""
	Generates 'numPlanes' number of slicing planes with origins directly
	above the originCSys. Note that this generates slicing planes in the Z Direction
	only, and for other axes of slicing, a transform and inverse transform should be 
	used to move the model in and out of the correct coordinate system.
	"""
	planes = []
	zHeights = np.linspace(originCSys.Point[2], originCSys.Point[2]+maxHeight, numPlanes)

	for zHeight in zHeights:
		originPoint = copy.deepcopy(originCSys.Point)
		originPoint[2] = zHeight
		planes.append(Plane(originPoint, [0, 0, 1]))
	# print(planes)
	return planes

def createMinMaxList( meshTopology ):
	minMaxList = []
	for face in meshTopology.faces:
		triangle = getTrianglefromIndex( meshTopology, face )
		minMaxList.append(findMinMaxZ(triangle))
	return minMaxList

def contourConstruction( segmentList ):
	segMap = {}
	contours = []
	# lines = []
	# cleanedSegmentList = []
	# for segment in segmentList:
	# 	print("%s, %s" % (segment[0].__hash__(), segment[1].__hash__()))
	# 	if segment[0].__hash__() != segment[1].__hash__():
	# 		cleanedSegmentList.append(segment)
	
	# segmentList = cleanedSegmentList
		# print(segment[1].__hash__())
	# 	p1 = segment[0].Coordinates[0:2]
	# 	p2 = segment[1].Coordinates[0:2]
	# 	print(p1)
	# 	print(p2)
	# 	lines.append([p1, p2])
	# lc = mc.LineCollection(lines, linewidths=2)
	# fig, ax = pl.subplots()
	# ax.add_collection(lc)
	# ax.autoscale()
	# ax.margins(0.1)
	# pl.show()
	# return False
	for segment in segmentList:
		if segment[0].__hash__() != segment[1].__hash__():
			continue
		u = segment[0]
		v = segment[1]
		if u not in segMap:
			segMap[u] = [v, None]
		else:
			segMap[u] = [segMap[u][0], v]

		if v not in segMap:
			segMap[v] = [u, None]
		else:
			segMap[v] = [segMap[v][0], u]

	while segMap:
		points = []
		points.append(segMap.keys()[0])
		[p, last] = segMap.pop(points[0])
		points.append(p)
		# print(segMap)
		j = 1
		while points[j-1].__hash__() != last.__hash__():
			[u, v] = segMap.pop(points[j])
			if tolerantEquals(u.Coordinates, points[j-1].Coordinates):
				points.append(v)
			else:
				points.append(u)
			j += 1
		contours.append(points)
		# print(len(contours[0]))
		# print(len(segMap.keys()))
		
	# print(len(contours))
	return contours


	# print(segMap.values())

def incrementalSlicing(meshTopology, planes, planeDelta):
	triangleLists = splitTriangles( meshTopology, planes, planeDelta)
	activeTriangles = set()
	segments = [[] for i in range(len(planes))]

	for i in range(len(planes)):
		activeTriangles.update(triangleLists[i])
		for triangle in copy.deepcopy(activeTriangles):
			if triangle.maxZ < planes[i].Point[2]:
				activeTriangles.remove(triangle)
			else:
				intPoints = calcPlaneTriangleIntersection( planes[i], getTrianglefromIndex(meshTopology, triangle.Indices))
				segments[i].append([Vertex(intPoints[0]), Vertex(intPoints[1])])

	return segments


def splitTriangles( meshTopology, planes, planeDelta):
	triangleLists = [set() for i in range(len(planes)+1)]
	if planeDelta > 0:
		for i, face in enumerate(meshTopology.faces, 0):
			triangle = getTrianglefromIndex( meshTopology, face )
			minT, maxT = meshTopology.minMaxList[i]
			
			if minT < planes[0].Point[2]:
				j = 0

			elif minT > planes[-1].Point[2]:
				j = len(planes)
			else:
				j = int(np.floor((minT-planes[0].Point[2])/planeDelta) +1 )
			triangleLists[j] = triangleLists[j].union([Face(face, minT, maxT)])

	return triangleLists

class Object(object):
	"""Describes an object to machine. Contins information about topology,
	mesh and voxel representation of what has been machined"""
	def __init__(self, mesh=None, meshTopology=None, voxelData=None):
		super(Object, self).__init__()
		self.mesh = mesh
		self.meshTopology = meshTopology
		self.voxelData = voxelData
		self.CSys = CSys()

class Face(object):
	"""
	A hashable Face, consisting of the attribute Indices which 
	is a list of three indices refering to a vertex list.
	"""
	def __init__(self, Indices, minZ=None, maxZ=None):
		super(Face, self).__init__()
		self.Indices = Indices
		self.minZ = minZ
		self.maxZ = maxZ
		
	def __hash__(self):
		return self.Indices[0] + self.Indices[1]*10**6 + self.Indices[2]*10**12

	def __eq__(self, other):
		if self.Indices[0] == other.Indices[0] and self.Indices[1] == other.Indices[1] and self.Indices[2] == other.Indices[2]:
			return True
		return False

	def __repr__(self):
		return "Face %s" % self.Indices

class Segment(object):
	"""
	A hashable Segment, consisting of the attributes P1 and P2 which 
	each are endpoints of a segment.The vertices are rounded to precision
	decimal places.
	"""
	def __init__(self, P1, P2, precision = 5):
		super(Segment, self).__init__()
		self.P1 = np.round(P1, precision)
		self.P2 = np.round(P2, precision)
		self.precision = precision

	def __hash__(self):
		return (self.P1[0]*10**self.precision + self.P1[1]*10**self.precision + self.P1[2]*10**self.precision
			  + self.P2[0]*10**self.precision + self.P2[1]*10**self.precision + self.P2[2]*10**self.precision)

	def __eq__(self, other):
		return tolerantEquals(self.P1, other.P1) and tolerantEquals(self.P2, other.P2)

	def __repr__(self):
		return "Segment -- P1 : %s, P2 : %s" % (self.P1, self.P2)

class Vertex(object):
	"""
	A hashable Vertex, consisting of the attribute Coordinates which 
	is a list of x, y, z coordinates for one vertex. The coordinatess are rounded to precision
	decimal places.
	"""
	def __init__(self, Coordinates, precision = 5):
		super(Vertex, self).__init__()
		self.Coordinates = np.round(Coordinates, precision)
		self.precision = precision

	def __hash__(self):
		return int(np.abs(self.Coordinates[0]*10**self.precision*10**(15-self.precision) + self.Coordinates[1]*10**self.precision + self.Coordinates[2]*10**self.precision))

	def __eq__(self, other):
		return tolerantEquals(self.Coordinates, other.Coordinates)

	def __repr__(self):
		return "Vertex -- Coordinates: %s" % (self.Coordinates)


class CSys(object):
	"""Describes a coordinate system in 3d space"""
	def __init__(self, Point=[0,0,0], Normal=[0,0,1], Xaxis=[1,0,0]):
		super(CSys, self).__init__()
		self.Point = Point
		self.Normal = Normal
		self.Xaxis = Xaxis

class Plane(object):
	"""Describes a Plane"""
	def __repr__(self):
		return ("Plane (Origin Point: %s Normal: %s)" % (self.Point, self.Normal))

	def __str__(self):
		return ("Plane -- Origin Point: %s Normal: %s" % (self.Point, self.Normal))

	def __init__(self, Point, Normal):
		super(Plane, self).__init__()
		self.Point = Point
		self.Normal = Normal

class Line(object):
	"""Describes a Line"""
	def __init__(self, Point, Direction):
		super(Line, self).__init__()
		self.Point = Point
		self.Direction = Direction

class MeshTopology(object):
	"""A topological description of a mesh"""
	def __init__(self, faces, vertices):
		super(MeshTopology, self).__init__()
		self.faces = faces
		self.vertices = vertices
		self.minMaxList = createMinMaxList(self)

def main():
	# Create a new plot
	figure = pyplot.figure()
	axes = mplot3d.Axes3D(figure)

	# Load the STL files and add the vectors to the plot
	meshData = mesh.Mesh.from_file('shellLower.stl')

	triangle = meshData.vectors[10]
	plane = Plane([0, .53434, 0], [.4, 1, 0])
	# print(np.max(meshData.vectors[:, :, 2]))

	maxHeight = np.max(meshData.vectors[:, :, 2])
	minHeight = np.min(meshData.vectors[:, :, 2])
	maxWidth = np.max(meshData.vectors[:, :, 1])
	minWidth = np.min(meshData.vectors[:, :, 1])
	maxLength = np.max(meshData.vectors[:, :, 0])
	minLength = np.min(meshData.vectors[:, :, 0])

	#Length, width, and height of the block to machine the part out of.
	blockDimensions = [np.abs(maxLength-minLength) + 1, np.abs(maxWidth-minWidth) + 1,np.abs(maxHeight-minHeight) -1 ] 

	slicePlanes = genSlicePlanes(blockDimensions[2], 10, CSys([0,0, minHeight], [0,0,1], [1,0,0]))

	# for triangle in meshData.vectors:
	# 	intPoints = calcPlaneTriangleIntersection( plane, triangle )
		# print(intPoints)

	# print(tolerantLinearSearch([-14.885533333333335, 5.0, -21.801099532671785], meshData.vectors))
	# aList = [0,3,5,6,7,8,10]
	# print(binary_search(aList, 10))
	meshTopology = genTopology(meshData)
	segments = incrementalSlicing( meshTopology, slicePlanes, np.abs(slicePlanes[0].Point[2]-slicePlanes[1].Point[2]))
	contours = contourConstruction(segments[1])

	targetObject = Object(meshData, meshTopology)

	# loop = genClosedLoop(meshTopology, plane)
	# # print(loop)

	# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(meshData.vectors))

	# # Auto scale to the mesh size
	# scale = meshData.points.flatten(-1)
	# axes.auto_scale_xyz(scale, scale, scale)

	# pyplot.show()

if __name__ == '__main__':
	main()
