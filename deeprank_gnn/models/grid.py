"""
This module holds the classes that are used when working with a 3D grid.
"""


from enum import Enum
from typing import Dict

import numpy
import h5py
import itertools
from scipy.signal import bspline

from ..domain.storage import (
    HDF5KEY_GRID_POINTS,
    HDF5KEY_GRID_X,
    HDF5KEY_GRID_Y,
    HDF5KEY_GRID_Z,
    HDF5KEY_GRID_CENTER,
    HDF5KEY_GRID_MAPPEDFEATURES,
    HDF5KEY_GRID_MAPPEDFEATURESVALUE
    )


class MapMethod(Enum):
    """ This holds the value of either one of 4 grid mapping methods.
        A mapping method determines how feature point values are divided over the grid points.
    """

    GAUSSIAN = 1
    FAST_GAUSSIAN = 2
    BSP_LINE = 3
    NEAREST_NEIGHBOURS = 4


class GridSettings:
    """ Objects of this class hold the settings to build a grid.
        The grid is basically a multi-divided 3D cube with
        the following properties:
         - points_count: the number of points on one edge of the cube
         - size: the length in Å of one edge of the cube
         - resolution: the size in Å of one edge subdivision. Also the distance between two points on the edge.
    """

    def __init__(self, points_count: int, size: float):
        self._points_count = points_count
        self._size = size

    @property
    def resolution(self) -> float:
        return self._size / self._points_count

    @property
    def size(self) -> float:
        return self._size

    @property
    def points_count(self) -> int:
        return self._points_count


class Grid:
    def __init__(self, id_:str, settings: GridSettings, center: numpy.array):
        """ An instance of this class holds everything that the grid is made of:
            - coordinates of points
            - names of features
            - feature values on each point
        """
        self.id = id_

        self._settings = settings
        self._center = center

        self._set_mesh(settings, center)

        self._features = {}

    def _set_mesh(self, settings: GridSettings, center: numpy.array):
        "builds the grid points"

        half_size = settings.size / 2

        min_x = center[0] - half_size
        max_x = center[0] + half_size
        self._xs = numpy.linspace(min_x, max_x, num=settings.points_count)

        min_y = center[1] - half_size
        max_y = center[1] + half_size
        self._ys = numpy.linspace(min_y, max_y, num=settings.points_count)

        min_z = center[2] - half_size
        max_z = center[2] + half_size
        self._zs = numpy.linspace(min_z, max_z, num=settings.points_count)

        self._ygrid, self._xgrid, self._zgrid = numpy.meshgrid(
            self._ys, self._xs, self._zs
        )

    @property
    def xs(self) -> numpy.array:
        return self._xs

    @property
    def xgrid(self) -> numpy.array:
        return self._xgrid

    @property
    def ys(self) -> numpy.array:
        return self._ys

    @property
    def ygrid(self) -> numpy.array:
        return self._ygrid

    @property
    def zs(self) -> numpy.array:
        return self._zs

    @property
    def zgrid(self) -> numpy.array:
        return self._zgrid

    @property
    def center(self) -> numpy.array:
        return self._center

    @property
    def features(self) -> Dict[str, numpy.array]:
        return self._features

    def add_feature_values(self, feature_name: str, data: numpy.ndarray):
        """ Makes sure feature values per grid point get stored.

            This method may be called repeatedly to add on to existing grid point values.
        """

        if feature_name not in self._features:
            self._features[feature_name] = data
        else:
            self._features[feature_name] += data

    def _get_mapped_feature_gaussian(
        self, position: numpy.ndarray, value: float
    ) -> numpy.ndarray:

        beta = 1.0

        fx, fy, fz = position
        distances = numpy.sqrt(
            (self.xgrid - fx) ** 2 + (self.ygrid - fy) ** 2 + (self.zgrid - fz) ** 2
        )

        return value * numpy.exp(-beta * distances)

    def _get_mapped_feature_fast_gaussian(
        self, position: numpy.ndarray, value: float
    ) -> numpy.ndarray:

        beta = 1.0
        cutoff = 5.0 * beta

        fx, fy, fz = position
        distances = numpy.sqrt(
            (self.xgrid - fx) ** 2 + (self.ygrid - fy) ** 2 + (self.zgrid - fz) ** 2
        )

        data = numpy.zeros(distances.shape)

        data[distances < cutoff] = value * numpy.exp(
            -beta * distances[distances < cutoff]
        )

        return data


    def _get_mapped_feature_bsp_line(
        self, position: numpy.ndarray, value: float
    ) -> numpy.ndarray:

        order = 4

        fx, fy, fz = position
        bsp_data = (bspline((self.xgrid - fx) / self.resolution, order) *
                    bspline((self.ygrid - fy) / self.resolution, order) *
                    bspline((self.zgrid - fz) / self.resolution, order))

        return value * bsp_data

    def _get_mapped_feature_nearest_neighbour( # pylint: disable=too-many-locals
        self, position: numpy.ndarray, value: float
    ) -> numpy.ndarray:

        fx, fy, fz = position

        distances_x = numpy.abs(self.xs - fx)
        distances_y = numpy.abs(self.ys - fy)
        distances_z = numpy.abs(self.zs - fz)

        indices_x = numpy.argsort(distances_x)[:2]
        indices_y = numpy.argsort(distances_y)[:2]
        indices_z = numpy.argsort(distances_z)[:2]

        sorted_x = distances_x[indices_x]
        weights_x = sorted_x / numpy.sum(sorted_x)

        sorted_y = distances_y[indices_y]
        weights_y = sorted_y / numpy.sum(sorted_y)

        sorted_z = distances_z[indices_z]
        weights_z = sorted_z / numpy.sum(sorted_z)

        indices = [indices_x, indices_y, indices_z]
        points = list(itertools.product(*indices))

        weight_products = list(itertools.product(weights_x, weights_y, weights_z))
        weights = [numpy.sum(p) for p in weight_products]

        neighbour_data = numpy.zeros(
            (self.xs.shape[0], self.ys.shape[0], self.zs.shape[0])
        )

        for point_index, point in enumerate(points):
            weight = weights[point_index]

            neighbour_data[point] = weight * value

        return neighbour_data

    def map_feature(
        self,
        position: numpy.ndarray,
        feature_name: str,
        feature_value: numpy.ndarray,
        method: MapMethod,
    ):
        "Maps point feature data at a given position to the grid, using the given method."

        for index, value in enumerate(feature_value):

            index_name = f"{feature_name}_{index:03d}"

            if method == MapMethod.GAUSSIAN:
                grid_data = self._get_mapped_feature_gaussian(position, value)

            elif method == MapMethod.FAST_GAUSSIAN:
                grid_data = self._get_mapped_feature_fast_gaussian(position, value)

            elif method == MapMethod.BSP_LINE:
                grid_data = self._get_mapped_feature_bsp_line(position, value)

            elif method == MapMethod.NEAREST_NEIGHBOUR:
                grid_data = self._get_mapped_feature_nearest_neighbour(position, value)

            # set to grid
            self.add_feature_values(index_name, grid_data)

    def write_to_hdf5(self, hdf5_path: str):
        "Write the grid data to hdf5, according to deeprank standards."

        with h5py.File(hdf5_path, "a") as hdf5_file:

            # create a group to hold everything
            grid_group = hdf5_file.require_group(self.id)

            try:
                # store grid points
                points_group = grid_group.create_group(HDF5KEY_GRID_POINTS)
                points_group.create_dataset(HDF5KEY_GRID_X, data=self.xs)
                points_group.create_dataset(HDF5KEY_GRID_Y, data=self.ys)
                points_group.create_dataset(HDF5KEY_GRID_Z, data=self.zs)
                points_group.create_dataset(HDF5KEY_GRID_CENTER, data=self.center)

                # store grid features
                features_group = grid_group.require_group(HDF5KEY_GRID_MAPPEDFEATURES)
                for feature_name, feature_data in self.features.items():

                    feature_group = features_group.require_group(feature_name)
                    feature_group.create_dataset(
                        HDF5KEY_GRID_MAPPEDFEATURESVALUE,
                        data=feature_data,
                        compression="lzf",
                        chunks=True,
                    )
            except:
                # do not leave behind a partly filled hdf5 entry
                del hdf5_file[self.id]
                raise
