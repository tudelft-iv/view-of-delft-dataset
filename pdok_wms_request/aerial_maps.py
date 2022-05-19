
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .tools import centre_to_bbox_coords, bbox_to_centre_coords, get_aspect_ratio, get_edges_distance, is_coord_in_bbox
from .transforms import CoordTransformer

# Website to get bbox values: http://bboxfinder.com/#51.997908,4.340329,52.022765,4.390283


class AerialMap:
    def __init__(self, map_, bbox, resolution):
        self.map_ = map_ 
        self.bbox = bbox
        self.centre, self.width, self.height = bbox_to_centre_coords(bbox)
        self.resolution = resolution    # in m/pixel
        self.coord_transformer = CoordTransformer()

    @property
    def bbox_nl(self):
        return [list(self.coord_transformer.t_nl_global(*coords)) for coords in self.bbox]

    def get_coordinate_from_pixel(self, x, y):
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError('Pixel coordinates must be of type int')

        if not is_coord_in_bbox((x, y), [[0,0], self.map_.shape[:2]]):
            raise ValueError('Pixel does not lie in map image')

        coord_lon = (self.resolution * x) + self.bbox_nl[0][0]
        coord_lat = (self.resolution * y) + self.bbox_nl[0][1] 

        return self.coord_transformer.t_global_nl(coord_lon, coord_lat)

    def get_pixel_from_coordinate(self, lon, lat):
        lon_nl, lat_nl = self.coord_transformer.t_nl_global(lon, lat)
        if not is_coord_in_bbox((lon_nl, lat_nl), self.bbox_nl):
            raise ValueError('Coordinates do not lie in bounding box')

        bbox_1_lon, bbox_1_lat = self.bbox_nl[0] 
        pixel_x = int((lon_nl - bbox_1_lon) / self.resolution)
        pixel_y = int((lat_nl - bbox_1_lat) / self.resolution)

        return pixel_x, pixel_y

    def show(self, mark_center=False):
        plt.imshow(self.map_)
        if mark_center:
            plt.gca().add_patch(plt.Circle((self.map_.shape[0] // 2, self.map_.shape[1] // 2), 100, color='r'))

        plt.show()



class AerialMapRetriever:
    def __init__(self,
            server_url="https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            "&FORMAT=image/png&TRANSPARENT=true&LAYERS=2020_ortho25&STYLES=&CRS=EPSG:28992",
            resolution=0.):
        self.server_url = server_url 
        self.resolution = resolution    # in m/pixel (0. for max)

        self.coord_transformer = CoordTransformer()

        # these are given by PDOK TODO make parameters 
        # note: these are actually listed as 5000, but this leads to an 'Internal Server Error'
        self.max_width = 4000
        self.max_height = 4000

    def _get_max_width_height(self, aspect_ratio):
        if aspect_ratio >= 1.:
            width = self.max_width 
            height = int(width / aspect_ratio)
        else:
            height = self.max_height
            width = int(height * aspect_ratio)
        return width, height

    def _get_pixels_from_resolution(self, bbox, resolution, aspect_ratio):
        if resolution is None:
            if self.resolution is None:
                raise ValueError('Resolution unspecified.')

            resolution = self.resolution

        if resolution == 0.:
            edges_pixels = self._get_max_width_height(aspect_ratio)
        else:
            edges_m = get_edges_distance(*[coord for coords in bbox for coord in coords])
            edges_pixels = [int(edge_m / resolution) for edge_m in edges_m]
        return edges_pixels

    def _get_pixels(self, bbox, x_pixels, y_pixels, resolution):
        aspect_ratio = get_aspect_ratio(*[coord for coords in bbox for coord in coords])
        if x_pixels is None and y_pixels is None:
            x_pixels, y_pixels = self._get_pixels_from_resolution(bbox, resolution, aspect_ratio) 
        elif x_pixels is None:
            x_pixels = y_pixels * aspect_ratio
        elif y_pixels is None:
            y_pixels = x_pixels / aspect_ratio 

        return x_pixels, y_pixels

    def get_map_from_centre(self, centre, width, height, x_pixels=None, y_pixels=None, resolution=None, wh_in_m=False):
        if wh_in_m:
            centre_nl = self.coord_transformer.t_nl_global(*centre)
            bbox_nl = centre_to_bbox_coords(centre_nl, width, height)
            bbox = [self.coord_transformer.t_global_nl(*coords) for coords in bbox_nl]
        else:
            bbox = centre_to_bbox_coords(centre, width, height)
        
        return self.get_map_from_corners(bbox, x_pixels, y_pixels, resolution)

    def get_map_from_corners(self, bbox, x_pixels=None, y_pixels=None, resolution=None):
        bbox_nl = [list(self.coord_transformer.t_nl_global(*coords)) for coords in bbox]

        if x_pixels is None or y_pixels is None:
            x_pixels, y_pixels = self._get_pixels(bbox_nl, x_pixels, y_pixels, resolution)

        bbox_1_lon = bbox_nl[0][0]
        bbox_1_lat = bbox_nl[0][1]
        bbox_2_lon = bbox_nl[1][0]
        bbox_2_lat = bbox_nl[1][1]
        params = {'WIDTH': x_pixels, 'HEIGHT': y_pixels,
                  'BBOX': f'{bbox_1_lon},'
                          f'{bbox_1_lat},'
                          f'{bbox_2_lon},'
                          f'{bbox_2_lat}'}
        
        # TODO catch errors in meaningful way
        response = requests.get(self.server_url, params)

        #print(response.text)

        response.raise_for_status()
        with io.BytesIO(response.content) as f:
            #img = Image.open(f)
            #img.save('example_5cm.png')
            map_array = np.array(Image.open(f))

        width, _ = get_edges_distance(bbox_1_lon, bbox_1_lat, bbox_2_lon, bbox_2_lat)
        res = width / x_pixels 
        return AerialMap(map_array, bbox, res)


if __name__ == "__main__":
    #bbox_1_lon, bbox_1_lat = 4.340329, 51.997908
    #bbox_2_lon, bbox_2_lat = 4.390283, 52.022765
    #bbox_1_lon, bbox_1_lat = 4.356636834345822, 52.0110137446908
    #bbox_2_lon, bbox_2_lat = 4.357563530763736, 52.012185918730005 
    #bbox = [[bbox_1_lon, bbox_1_lat],
    #        [bbox_2_lon, bbox_2_lat]]

    #width, height = get_edges_distance(bbox_1_lon, bbox_1_lat, bbox_2_lon, bbox_2_lat)
    #centre = [bbox_1_lon + width/2, bbox_1_lat + height/2]
    lat = 52.0116
    lon = 4.3571
    centre = [lon, lat] 
    wh_in_m = True
    #width = 100
    #height = 200
    width = 70
    height = 70

    #y_length = 1000
    #y_length = 3000
    #y_length = None 
    #resolution = 1
    #resolution = 0.2
    #resolution = 2 
    resolution = 0.
    #resolution = 0.05
    #resolution = 0.075
    #resolution = 0.25
    #server_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&FORMAT=image/png&TRANSPARENT=true&LAYERS=2020_ortho25&STYLES=&CRS=EPSG:28992"
    server_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&FORMAT=image/png&TRANSPARENT=true&LAYERS=Actueel_orthoHR&STYLES=&CRS=EPSG:28992"
    
    map_retriever = AerialMapRetriever(server_url, resolution=resolution)

    #map_from_corners = map_retriever.get_map_from_corners(bbox)
    #map_from_corners.show()
    #pixel = map_from_corners.get_pixel_from_coordinate(bbox_1_lon, bbox_1_lat)
    #pixel = np.array(map_from_corners.get_pixel_from_coordinate(bbox_2_lon - 1e-8, bbox_2_lat - 1e-8)) - 1
    #coords = map_from_corners.get_coordinate_from_pixel(*pixel)
    #print(bbox_1_lon, bbox_1_lat)
    #print(pixel)
    #print(coords)
    #assert np.allclose(coords, (bbox_1_lon, bbox_1_lat))

    map_from_centre = map_retriever.get_map_from_centre(centre, width, height, resolution=resolution, wh_in_m=wh_in_m)
    map_from_centre.show()

    #assert np.all(from_corners == from_centre)

