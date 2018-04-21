import numpy as np
import copy
import objectDefinitions as oD
from shapely.ops import nearest_points, cascaded_union
from sortedcontainers import SortedList, SortedSet, SortedDict
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box, MultiPoint
import plotContours
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

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

	L1 = oD.Line(triangle[triTip], intVecs[0])
	L2 = oD.Line(triangle[triTip], intVecs[1])

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

	return oD.MeshTopology(faceList, vertexList)


def genClosedLoop( MeshTopology,  plane):
	"""generates a closed loop from the intersection of a plane with a mesh"""
	allIntPoints = []
	vertexList = SortedList(key=cmp_to_key(tolerantCompare))
	pairList = SortedList()

	for i in MeshTopology.faces:
		triangle = MeshTopology.getTrianglefromIndex(i)
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

	d = meshTopology.getTrianglefromIndex(finalList)

	return d


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
Converts a polygon to a list. If the polygon is a multipolygon or already a list
it breaks it into a list of Polygon objects. If not, it returns a list containing the element
poly.
"""
def polyToList( poly ):
	
	if (type(poly) == list) or poly.geom_type == "MultiPolygon" :
		polyList = []
		for internalPoly in poly:
			polyList.append(Polygon(internalPoly.exterior.coords))
			for interior in internalPoly.interiors:
				polyList.append(Polygon(interior.coords))
		return polyList
	else:
		polyList = []
		for interior in poly.interiors:
			polyList.append(Polygon(interior.coords))

		if polyList:
			polyList.append(Polygon(poly.exterior.coords))
			return polyList
		else:
			return [poly]

"""
Optional paramaters to be added later for toolpath generation.
Units are in Millimeters
"""
def genToolPath( multiPoly, pathStepSize=3, initial_Offset=0, plot=False, zHeight=0, topBounds=0 ):

	polyList = []
	step = initial_Offset
	inc = 0
	while (1):
		inc += 1
		poly = multiPoly.buffer(step)
		if poly.is_empty:
			break

		polyList.append(polyToList(poly))
		step = step-pathStepSize

	orderedPolyList = []
	orderedPolyList.append(polyList[0].pop(0))

	currPolyIndex = 0
	while (1):
		if not polyList:
			break

		if currPolyIndex + 1 == len(polyList):
			if not polyList[0]:
				polyList.pop(0)
				currPolyIndex = currPolyIndex - 1
				continue
			orderedPolyList.append(polyList[0].pop(0))
			currPolyIndex = 0
			continue

		for i, nextPoly in enumerate(polyList[currPolyIndex+1], 0):
			if nextPoly.within(orderedPolyList[-1]):
				orderedPolyList.append(polyList[currPolyIndex+1].pop(i))
				break
		currPolyIndex = currPolyIndex + 1

	# print(orderedPolyList[0].exterior.coords[0])
	x = np.array([])
	y = np.array([])
	z = np.array([])
	for i, poly in enumerate(orderedPolyList[1::], 1):

		x_, y_ = orderedPolyList[i-1].exterior.coords.xy
		x = np.concatenate([x ,x_])if x.shape else x_
		y = np.concatenate([y ,y_])if y.shape else y_
		z = np.concatenate([z, np.full_like(x_, zHeight)])
		closestIndex = findClosestPolyIndex(orderedPolyList[i].exterior.coords, 
											orderedPolyList[i-1].exterior.coords[0])
		orderedPolyList[i] = Polygon(orderedPolyList[i].exterior.coords[closestIndex:-1] +
									 orderedPolyList[i].exterior.coords[0:closestIndex])
		changePath = LineString([orderedPolyList[i-1].exterior.coords[0], orderedPolyList[i].exterior.coords[0]])

		if changePath.crosses(multiPoly):

			cx, cy = changePath.coords.xy
			cz = np.full_like(cx, topBounds)
			x = np.concatenate([x, cx])
			y = np.concatenate([y, cy])
			z = np.concatenate([z, cz])

	x_, y_ = orderedPolyList[-1].exterior.coords.xy
	x = np.concatenate([x ,x_])
	y = np.concatenate([y ,y_])
	z = np.concatenate([z, np.full_like(x_, zHeight)])

	# polyList = orderedPolyList
	# print(polyList)
	# cmbd = plotContours.combinePolygons(objectPath)
	# ax = plotContours.drawPoly(cmbd, [])
	# 	# print(polyBuffer)

	# orderedPolyList.append(multiPoly)
	# res = plotContours.combinePolygons(orderedPolyList[0])
	res = lambda: None
	res.PolyList = orderedPolyList
	res.XYZ = [x, y, z]
	if plot:
		ax = plotContours.drawPoly(res.PolyList, [])
		# print(polyBuffer)

		ax.set_title('Polygon')
		pyplot.show()

	return res


def generateToolOffsets( contourList, zSpacing ):
	"""
	Takes in a contour list (a list of polygons) that already
	take overhangs into account (i.e., all n+m contours contain
	the contour n) and generates a new contour list, where the resulting
	contours prevent the tool from hitting previous layers. 
	"""

	# This assumes a cylindrical tool with a 90 degree conical tip.
	toolDiameter = 6.35

	maxOffset = toolDiameter/2

	resContourList = copy.deepcopy(contourList)
	for i, contour in enumerate(contourList, 0):
		currOffset = zSpacing
		buffList = []
		n = 1
		while currOffset < maxOffset and n <= i:
			buffList.append(contourList[i-n].buffer(currOffset))
			n = n + 1
			currOffset = zSpacing*n

		resContourList[i] = cascaded_union(buffList + [resContourList[i]])
		# if buffList:
		# 	resContourList[i] = buffList[-1]
	return resContourList


def genSlicePlanes( maxHeight, spacing, originCSys ):
	"""
	Generates a number of slicing planes with origins directly
	above the originCSys, and are 'spacing' distance apart from eachother. 
	Note that this generates slicing planes in the Z Direction
	only, and for other axes of slicing, a transform and inverse transform should be 
	used to move the model in and out of the correct coordinate system.
	"""

	planes = []
	# print(np.arange(originCSys.Point[2], originCSys.Point[2]+maxHeight, .8))
	zHeights = np.arange(originCSys.Point[2], originCSys.Point[2]+maxHeight, spacing)

	for zHeight in zHeights:
		originPoint = copy.deepcopy(originCSys.Point)
		originPoint[2] = zHeight
		planes.append(oD.Plane(originPoint, [0, 0, 1]))
	return planes

def contourConstruction( segmentList ):
	segMap = {}
	contours = []

	for segment in segmentList:
		if segment[0].__hash__() == segment[1].__hash__():
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

	# print(len(segMap))
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
				intPoints = calcPlaneTriangleIntersection( planes[i], meshTopology.getTrianglefromIndex(triangle.Indices))
				segments[i].append([oD.Vertex(intPoints[0]), oD.Vertex(intPoints[1])])

	return segments

def splitTriangles( meshTopology, planes, planeDelta):
	triangleLists = [set() for i in range(len(planes)+1)]
	if planeDelta > 0:
		for i, face in enumerate(meshTopology.faces, 0):
			triangle = meshTopology.getTrianglefromIndex( face )
			minT, maxT = meshTopology.minMaxList[i]
			
			if minT < planes[0].Point[2]:
				j = 0

			elif minT > planes[-1].Point[2]:
				j = len(planes)
			else:
				j = int(np.floor((minT-planes[0].Point[2])/planeDelta) +1 )
			triangleLists[j] = triangleLists[j].union([oD.Face(face, minT, maxT)])

 	return triangleLists

def findClosestPolyIndex( coords, point ):
	minDist = float("inf")
	minIndex = -1
	for i, coord in enumerate(coords, 0):
		newDist = dist(coord, point)
		if newDist < minDist:
			minIndex = i
			minDist = newDist
	# print(minIndex)
	return minIndex	

def dist(x,y):   
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def getPolyDepth( multiPoly ):
	"""
	Returns a list of polygons (not a MultiPolygon) where each polygon has the attribute
	'depth' which represents the depth of containment of the polygon
	"""
	multiPoly = MultiPolygon(sorted(multiPoly, key=oD.Within, reverse=True))

	depthMapping = []
	multiPolyList = []
	for poly in multiPoly:
		poly.depth = 0
		multiPolyList.append(poly)

	for i, poly in enumerate(multiPolyList, 0):
		if i == len(multiPolyList)-1:
			break
		if oD.Within(multiPolyList[i+1]) < oD.Within(poly):
			multiPolyList[i+1].depth = poly.depth + 1
		else:
			multiPolyList[i+1].depth = poly.depth

	polygonGrouping = []
	polyInteriors = []
	for poly in multiPolyList:
		# print(poly.depth)
		polyExterior = None

		if poly.depth % 2 == 0:
			polygonGrouping.append([poly, []])
			# print("Outside")
		else:
			polygonGrouping[-1][1].append(poly)
			# print("Inside")

	# print(polygonGrouping)
	polyResults = []
	for polygon in polygonGrouping:

		if polygon[1]:
			tupleList =[]
			for polygonInstance in polygon[1]:
				tupleList.append(list(polygonInstance.exterior.coords))
			polyResults.append(Polygon(list(polygon[0].exterior.coords), tupleList))

		else:
			polyResults.append(polygon[0])

	# print(Polygon(polygonGrouping[0][0], polygonGrouping[0][1]))
	# print(MultiPolygon(polyResults))
	return MultiPolygon(polyResults)
