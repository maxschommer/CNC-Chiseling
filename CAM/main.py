import numpy as np
import bintrees
from stl import mesh
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


def tolerantBinarySearchVertexList( vertex, vertexList ):
    """Performs iterative binary search to find the position of an integer in a given, sorted, list.
    vertexList -- sorted list of integers
    vertex -- integer you are searching for the position of
    """

    first = 0
    last = len(vertexList) - 1
    vertex = vertex.tolist()
    while first <= last:
        i = (first + last) / 2
        # print(vertex)
        if tolerantEquals(vertexList[i], vertex):
            return i
        elif vertexList[i] > vertex:
            last = i - 1
        elif vertexList[i] < vertex:
            first = i + 1
        else:
			return None

def tolerantEquals( v1, v2, tolerance=.00001 ):

	if (np.abs(v1[0]-v2[0]) < tolerance) and (np.abs(v1[1]-v2[1]) < tolerance) and (np.abs(v1[2]-v2[2]) < tolerance):
		return True
	else:
		return False

def calcPlaneLineIntersection( plane, line ):
	sI = np.dot(plane.Normal, np.subtract(plane.Point, line.Point))/np.dot(plane.Normal, line.Direction)
	return line.Point + sI * line.Direction

def pointPlaneDist( plane, point ):
	pass

def dotProduct( v1, v2 ):
	return v1.P1*v2.P1 + v1.P2*v2.P2 + v1.P3*v2.P3

def genTopology ( mesh ):

	vertexList = SortedList()
	faceList = SortedList()

	for triangle in mesh.vectors:
		for vertex in triangle:
			j = tolerantBinarySearchVertexList( vertex, vertexList )
			if (j == None):
				vertexList.add(vertex.tolist())


	for triangle in mesh.vectors:
		face = []
		for vertex in triangle:
			j = tolerantBinarySearchVertexList( vertex, vertexList )
			if (j != None):
				face.append(j)
			else:
				raise ValueError("Couldn't find vertex in list")
		faceList.add(face)
		

	adjacentFaceList = [0]*len(faceList)
	for j, face in enumerate(faceList):
		adjacentFaces = []
		for i, isAdjacentFace in enumerate(faceList):
			if j == i:
				continue
			if i in adjacentFaces:
				continue
			for vertex in face:
				if (vertex in isAdjacentFace):
					adjacentFaces.append(i)
					break

		adjacentFaceList[j] = adjacentFaces

	return MeshTopology(adjacentFaceList, faceList, vertexList)



def genClosedLoop( MeshTopology,  plane ):
	pass

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

class Plane(object):
	"""Describes a Plane"""
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
	def __init__(self, adjacentFaces, faces, vertices):
		super(MeshTopology, self).__init__()
		self.adjacentFaces = adjacentFaces
		self.faces = faces
		self.vertices = vertices

def main():
	# Create a new plot
	# figure = pyplot.figure()
	# axes = mplot3d.Axes3D(figure)

	# Load the STL files and add the vectors to the plot
	meshData = mesh.Mesh.from_file('cube.stl')

	triangle = [[0,1,0], [1,1,0], [.5,.4,0]]
	plane = Plane([0, .2, 0], [1, 1, 0])

	for triangle in meshData.vectors:
		intPoints = calcPlaneTriangleIntersection( plane, triangle )
		print(intPoints)
	# aList = [0,3,5,6,7,8,10]
	# print(binary_search(aList, 10))
	meshTopology = genTopology(meshData)
	genToolPath( [] )
	print(meshTopology.faces)
	# print(meshTopology.adjacentFaces)

	# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(meshData.vectors))

	# # Auto scale to the mesh size
	# scale = meshData.points.flatten(-1)
	# axes.auto_scale_xyz(scale, scale, scale)


	#ppyplot.show()

if __name__ == '__main__':
	main()
