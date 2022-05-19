
import numpy as np


def get_aspect_ratio(x_1, y_1, x_2, y_2):
    a, b = get_edges_distance(x_1, y_1, x_2, y_2)
    return a / b


def get_edges_distance(x_1, y_1, x_2, y_2):
    a = abs(x_2 - x_1)
    b = abs(y_2 - y_1)
    return a, b


def centre_to_bbox_coords(centre, width, height):
    return [[centre[0] - width/2, centre[1] - height/2],
            [centre[0] + width/2, centre[1] + height/2]] 


def bbox_to_centre_coords(bbox):
    width, height = get_edges_distance(*[coord for coords in bbox for coord in coords])
    centre = [bbox[0][0] + width/2, bbox[0][1] + height/2]
    return centre, width, height


def is_coord_in_bbox(coord, bbox):
    #print(coord)
    #print(bbox)
    #exit()
    bbox_array = np.array(bbox)
    lon_valid = coord[0] >= bbox_array[0,0] and coord[0] < bbox_array[1,0]
    lat_valid = coord[1] >= bbox_array[0,1] and coord[1] < bbox_array[1,1]

    return lon_valid and lat_valid

