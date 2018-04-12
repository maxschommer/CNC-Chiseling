import numpy as np
import processingFunctions as pF


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
		return pF.tolerantEquals(self.P1, other.P1) and tolerantEquals(self.P2, other.P2)

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
		return pF.tolerantEquals(self.Coordinates, other.Coordinates)

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
		self.minMaxList = self.createMinMaxList()
	def createMinMaxList( self ):
		minMaxList = []
		for face in self.faces:
			triangle = self.getTrianglefromIndex( face )
			minMaxList.append(self.findMinMaxZ(triangle))
		return minMaxList

	def getTrianglefromIndex (self , face ):
		"""Helper function for genClosedLoop. Returns vertices of triangles from a face"""
		triangle = []
		for i in face:
			triangle.append(self.vertices[i])
		return triangle

	def findMinMaxZ(self, triangle):
		triangle = np.asarray(triangle)
		return [np.min(triangle[:,2], 0), np.max(triangle[:,2], 0)]




class Within(object):
	"""
	Wrapper object that allows sorting of Shapely objects using
	the within operator
	"""
	def __init__(self, polygon):
		self.polygon = polygon
	def __lt__(self, other):
		return self.polygon.within(other.polygon)
