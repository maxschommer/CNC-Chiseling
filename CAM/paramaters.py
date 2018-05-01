import numpy as np

ZERO_OFFSET = np.array([415,-210,-150])  # The offset needed to zero to the material location
MV_BLOCK = True # If the block should be moved with the offset or not

Z_SPACING = .5 # Layer Height


 
BLOCK_CENTER = np.array([0, 0, 15]) # Center of the box pre-offset (in part coordinates)
BLOCK_DIMENSIONS = np.array([40, 40, 30])


SAFE_HEIGHT = 30 # Safe height above the block to move to to clear the part

PATH_INITIAL_OFFSET = 0 # The amount to offset the toolpaths from the sliced contour
PATH_STEP_SIZE = 2

TOOL_DIAMETER = 6.35 # Assumes a cylindrical tool with a 90 degree conical tip

BUFFER_RES = 6 # Resolution of the corner circles when a sharp angle is offset in toolpath generation
POLY_TOL = .4 # Tolerance of polygons generated for toolpaths.


