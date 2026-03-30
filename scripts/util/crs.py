import shapely
import geopandas as gpd


def change_crs(shape: shapely.Geometry, src: str, dst: str) -> shapely.Geometry:
    return gpd.GeoDataFrame({"geometry": [shape]}, crs=src).to_crs(dst).geometry[0]
