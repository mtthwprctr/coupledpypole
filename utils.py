import numpy as np

def pointy_hex_corner_edges(center, size, i):
    '''
    Method translated from https://www.redblobgames.com/grids/hexagons/
    '''

    angle_deg = 60 * i - 30
    angle_rad = np.pi / 180 * angle_deg 

    pos_x = center[0] + size * np.cos(angle_rad)
    pos_y = center[1] + size * np.sin(angle_rad)

    return np.array([pos_x, pos_y])

def flat_hex_corner_edges(center, size, i):
    '''
    Method translated from https://www.redblobgames.com/grids/hexagons/
    '''

    angle_deg = 60 * i
    angle_rad = np.pi / 180 * angle_deg

    pos_x = center[0] + size * np.cos(angle_rad)
    pos_y = center[1] + size * np.sin(angle_rad)

    return np.array([pos_x, pos_y])    

def square_edges(center, size, i):
    angle_deg = 90 * i - 45
    angle_rad = np.pi / 180 * angle_deg

    pos_x = center[0] + size * np.cos(angle_rad)
    pos_y = center[1] + size * np.sin(angle_rad)

    return np.array([pos_x, pos_y])    


def point_in_polygon(p, polygon):
    ''' Method for checking if point p is within polygon 
    translated from C version here https://stackoverflow.com/a/16391873 '''
    inside = False

    min_x = polygon[0][0]
    max_x = polygon[0][0]

    min_y = polygon[0][1]
    max_y = polygon[0][1]

    for vertex in polygon:
        min_x = min(vertex[0], min_x)
        max_x = min(vertex[0], max_x)
        min_y = min(vertex[1], min_y)
        max_y = min(vertex[1], max_y)

    if (p[0] < min_x, p[0] > max_x, p[1] < min_y, p[1] > max_y) == False:
        return False

    for i in np.arange(len(polygon)):
        j = i - 1
        if ((polygon[i][1] > p[1]) != (polygon[j][1] > p[1]) and 
                p[0] < (polygon[j][0] - polygon[i][0]) * (p[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
    return inside

def rot(angle, z=True):
    if z:
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])
    else:
        return np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
