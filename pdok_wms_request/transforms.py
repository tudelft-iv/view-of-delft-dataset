
from pyproj import Transformer


class CoordTransformer:
    def __init__(self):
        # Define transformers for local and global coordinate systems
        self.transformer_nl_global = Transformer.from_crs("epsg:4326", "epsg:28992", always_xy=True)
        self.transformer_global_nl = Transformer.from_crs("epsg:28992", "epsg:4326", always_xy=True)

    def t_nl_global(self, lon, lat):
        return self.transformer_nl_global.transform(lon, lat)

    def t_global_nl(self, lon, lat):
        return self.transformer_global_nl.transform(lon, lat)

