import numpy as np
import math
import pickle

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB (yaw,pitch,roll)

def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def writeVplus(inputFile, outputFile):

    data = pickle.load(open( inputFile, "rb" ) )

    with open(outputFile, 'w') as f:
        f.write('SPEED 0.5 ALWAYS\n')
        f.write('READY\n')
        f.write('DRIVE 1,-20,10\n')
        f.write('DRIVE 2,60,10\n')
        f.write('DRIVE 3,40,10\n')
        f.write('DRIVE 5,50,10\n')
        f.write('ABOVE\n')
        f.write('MOVE TRANS(430,-150,0,0,180,180)\n')
        for i in range(1000):
            f.write('MOVES TRANS({},{},{},0,180,180)\n'.format(data[0][i],data[1][i],data[2][i]))


if __name__ == '__main__':
    # R = eulerAnglesToRotationMatrix([3.14159, 1.5708, 0])
    # print(R)
    # eulerAngles = rotationMatrixToEulerAngles(R)
    # print(eulerAngles)

    writeVplus('out.txt','output.pg')