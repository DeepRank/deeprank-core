from enum import Enum
from typing import Callable, Union, List, Dict, Optional
from uuid import uuid4
import logging

import numpy
import h5py

from deeprank_gnn.models.structure import Atom, Residue
from deeprank_gnn.models.contact import Contact
from deeprank_gnn.models.grid import MapMethod, Grid, GridSettings
from deeprank_gnn.domain.storage import *


_log = logging.getLogger(__name__)


class Edge:
    def __init__(self, id_: Contact):
        self.id = id_
        self.features = {}

    def add_feature(self, feature_name: str, feature_function: Callable[[Contact], float]):
        feature_value = feature_function(self.id)

        self.features[feature_name] = feature_value

    @property
    def position1(self) -> numpy.array:
        return self.id.item1.position

    @property
    def position2(self) -> numpy.array:
        return self.id.item2.position

    def has_nan(self) -> bool:
        "whether there are any NaN values in the edge's features"

        for feature_data in self.features.values():
            if numpy.any(numpy.isnan(feature_data)):
                return True

        return False


class NodeType(Enum):
    ATOM = 1
    RESIDUE = 2


class Node:
    def __init__(self, id_: Union[Atom, Residue]):
        if type(id_) == Atom:
            self._type = NodeType.ATOM

        elif type(id_) == Residue:
            self._type = NodeType.RESIDUE
        else:
            raise TypeError(type(id_))

        self.id = id_

        self.features = {}

    @property
    def type(self):
        return self._type

    def has_nan(self) -> bool:
        "whether there are any NaN values in the node's features"

        for feature_data in self.features.values():
            if numpy.any(numpy.isnan(feature_data)):
                return True

        return False

    def add_feature(self, feature_name: str, feature_function: Callable[[Union[Atom, Residue]], numpy.ndarray]):
        feature_value = feature_function(self.id)

        if len(feature_value.shape) != 1:
            shape_s = 'x'.join(feature_value.shape)
            raise ValueError(f"Expected a 1-dimensional array for feature {feature_name}, but got {shape_s}")

        self.features[feature_name] = feature_value

    @property
    def position(self) -> numpy.array:
        return self.id.position


class Graph:
    def __init__(self, id_: str):
        self.id = id_

        self._nodes = {}
        self._edges = {}

        # targets are optional and may be set later
        self.targets = {}

    def add_node(self, node: Node):
        self._nodes[node.id] = node

    def get_node(self, id_: Union[Atom, Residue]) -> Node:
        return self._nodes[id_]

    def add_edge(self, edge: Edge):
        self._edges[edge.id] = edge

    def get_edge(self, id_: Contact) -> Edge:
        return self._edges[edge.id]

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> List[Node]:
        return list(self._edges.values())

    def has_nan(self) -> bool:
        "whether there are any NaN values in the graph's features"

        for node in self._nodes.values():
            if node.has_nan():
                return True

        for edge in self._edges.values():
            if edge.has_nan():
                return True

        return False

    def map_to_grid(self, grid: Grid, method: MapMethod):

        for edge in self._edges.values():
            for feature_name, feature_value in edge.features.items():
                grid.map_feature(edge.position1, feature_name, feature_value, method)
                grid.map_feature(edge.position2, feature_name, feature_value, method)

        for node in self._nodes.values():
            for feature_name, feature_value in node.features.items():
                grid.map_feature(node.position, feature_name, feature_value, method)

    def write_to_hdf5(self, hdf5_path: str):
        "Write a featured graph to an hdf5 file, according to deeprank standards."

        with h5py.File(hdf5_path, 'a') as hdf5_file:

            # create a group to hold everything
            graph_group = hdf5_file.require_group(self.id)

            # store node names
            node_names = numpy.array([str(key) for key in self._nodes]).astype('S')
            graph_group.create_dataset(HDF5KEY_GRAPH_NODENAMES, data=node_names)

            # store node features
            node_features_group = graph_group.create_group(HDF5KEY_GRAPH_NODEFEATURES)
            node_key_list = list(self._nodes.keys())
            first_node_data = list(self._nodes.values())[0].features
            node_feature_names = list(first_node_data.keys())
            for node_feature_name in node_feature_names:

                node_feature_data = [node.features[node_feature_name] for node in self._nodes.values()]

                node_features_group.create_dataset(node_feature_name, data=node_feature_data)

            # store edges
            edge_indices = []
            edge_names = []

            first_edge_data = list(self._edges.values())[0].features
            edge_feature_names = list(first_edge_data.keys())

            edge_feature_data = {name: [] for name in edge_feature_names}

            for edge_id, edge in self._edges.items():

                id1, id2 = edge_id
                node_index1 = node_key_list.index(id1)
                node_index2 = node_key_list.index(id2)

                edge_indices.append((node_index1, node_index2))
                edge_names.append(f"{id1}-{id2}")

                for edge_feature_name in edge_feature_names:
                    edge_feature_data[edge_feature_name].append(edge.features[edge_feature_name])

            graph_group.create_dataset(HDF5KEY_GRAPH_EDGENAMES, data=numpy.array(edge_names).astype('S'))

            graph_group.create_dataset(HDF5KEY_GRAPH_EDGEINDICES, data=edge_indices)

            edge_feature_group = graph_group.create_group(HDF5KEY_GRAPH_EDGEFEATURES)
            for edge_feature_name in edge_feature_names:
                edge_feature_group.create_dataset(edge_feature_name, data=edge_feature_data[edge_feature_name])

            # store target values
            score_group = graph_group.create_group(HDF5KEY_GRAPH_SCORE)
            for target_name, target_data in self.targets.items():
                score_group.create_dataset(target_name, data=target_data)

    def write_as_grid_to_hdf5(self, hdf5_path: str, settings: GridSettings, method: MapMethod) -> str:

        center = numpy.mean([node.position for node in self._nodes], axis=0)
        grid = Grid(self.id, settings, center)

        self.map_to_grid(grid, method)
        grid.to_hdf5(hdf5_path)

        return hdf5_path
