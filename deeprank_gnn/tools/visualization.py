import logging
from copy import deepcopy
from typing import Optional
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import h5py
import numpy
import networkx
import community
import markov_clustering

from deeprank_gnn.tools.embedding import manifold_embedding
from deeprank_gnn.domain.feature import (
    FEATURENAME_POSITION,
    FEATURENAME_CHAIN,
    FEATURENAME_EDGETYPE,
    EDGETYPE_INTERNAL,
    EDGETYPE_INTERFACE
)
from deeprank_gnn.domain.storage import (
    HDF5KEY_GRAPH_NODENAMES,
    HDF5KEY_GRAPH_NODEFEATURES,
    HDF5KEY_GRAPH_EDGENAMES,
    HDF5KEY_GRAPH_EDGEINDICES,
    HDF5KEY_GRAPH_EDGEFEATURES
)


_log = logging.getLogger(__name__)


def _get_node_key(value):
    if isinstance(value, str):
        return value

    key = ""
    for item in value:
        if isinstance(item, (bytes, numpy.bytes_)):
            key = item.decode()

        elif isinstance(item, str):
            key += item

        else:
            key += str(item)

    return key


def hdf5_to_networkx(graph_group: h5py.Group) -> networkx.Graph: # pylint: disable=too-many-locals
    """Read a hdf5 group into a networkx graph"""

    graph = networkx.Graph()

    # read nodes
    node_names = [
        _get_node_key(key) for key in graph_group[HDF5KEY_GRAPH_NODENAMES][()]
    ]
    node_features_group = graph_group[HDF5KEY_GRAPH_NODEFEATURES]
    node_features = {}
    node_feature_names = list(node_features_group.keys())
    for node_feature_name in node_feature_names:
        node_features[node_feature_name] = node_features_group[node_feature_name][()]

    for node_index, node_name in enumerate(node_names):
        graph.add_node(node_name)
        for node_feature_name in node_feature_names:
            graph.nodes[node_name][node_feature_name] = node_features[
                node_feature_name
            ][node_index]

    # read edges
    edge_names = graph_group[HDF5KEY_GRAPH_EDGENAMES][()]
    edge_node_indices = graph_group[HDF5KEY_GRAPH_EDGEINDICES][()]
    edge_features_group = graph_group[HDF5KEY_GRAPH_EDGEFEATURES]
    edge_features = {}
    edge_feature_names = list(edge_features_group.keys())
    for edge_feature_name in edge_feature_names:
        edge_features[edge_feature_name] = edge_features_group[edge_feature_name][()]

    for edge_index, _ in enumerate(edge_names):
        node1_index, node2_index = edge_node_indices[edge_index]
        node1_name = node_names[node1_index]
        node2_name = node_names[node2_index]
        edge_key = (node1_name, node2_name)

        graph.add_edge(node1_name, node2_name)
        graph.edges[node1_name, node2_name][FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE
        for edge_feature_name in edge_feature_names:
            graph.edges[edge_key][edge_feature_name] = edge_features[edge_feature_name][
                edge_index
            ]

    return graph


def _get_edge_type_name(value):
    if isinstance(value, (bytes, numpy.bytes_)):

        return value.decode()

    return value


def plotly_2d( # noqa
    graph: networkx.Graph,
    out: Optional[str] = None,
    offline: bool = False,
    iplot: bool = True,
    disable_plot: bool = False,
    method: str = "louvain",
):
    """Plots the interface graph in 2D"""

    if offline:
        import plotly.offline as py # pylint: disable=import-outside-toplevel
    else:
        import chart_studio.plotly as py # pylint: disable=import-outside-toplevel

    pos = numpy.array(
        [v.tolist() for _, v in networkx.get_node_attributes(graph, "pos").items()]
    )
    pos2D = manifold_embedding(pos)
    dict_pos = dict(zip(graph.nodes, pos2D))
    networkx.set_node_attributes(graph, dict_pos, "pos2D")

    # remove interface edges for clustering
    gtmp = deepcopy(graph)
    ebunch = []
    for e in graph.edges:
        typ = graph.edges[e][FEATURENAME_EDGETYPE]
        if isinstance(typ, bytes):
            typ = typ.decode("utf-8")
        if typ == EDGETYPE_INTERFACE:
            ebunch.append(e)
    gtmp.remove_edges_from(ebunch)

    if method == "louvain":
        cluster = community.best_partition(gtmp)

    elif method == "mcl":
        matrix = networkx.to_scipy_sparse_matrix(gtmp)
        # run MCL with default parameters
        result = markov_clustering.run_mcl(matrix)
        mcl_clust = markov_clustering.get_clusters(result)  # get clusters
        cluster = {}
        node_key = list(graph.nodes.keys())
        for ic, c in enumerate(mcl_clust):
            for node in c:
                cluster[node_key[node]] = ic

    # get the colormap for the clsuter line
    ncluster = numpy.max([v for _, v in cluster.items()]) + 1
    cmap = plt.cm.nipy_spectral
    N = cmap.N
    cmap = [cmap(i) for i in range(N)]
    cmap = cmap[:: int(N / ncluster)]
    cmap = "plasma"

    edge_trace_list, internal_edge_trace_list = [], []

    node_connect = {}
    for edge in graph.edges:

        edge_type = _get_edge_type_name(graph.edges[edge[0], edge[1]]["type"])
        if edge_type == EDGETYPE_INTERNAL:
            trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter.Line(color="rgb(110,110,110)", width=3),
            )

        elif edge_type == EDGETYPE_INTERFACE:
            trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter.Line(color="rgb(210,210,210)", width=1),
            )
        else:
            continue

        x0, y0 = graph.nodes[edge[0]]["pos2D"]
        x1, y1 = graph.nodes[edge[1]]["pos2D"]

        trace["x"] += (x0, x1, None)
        trace["y"] += (y0, y1, None)

        if edge_type == EDGETYPE_INTERNAL:
            internal_edge_trace_list.append(trace)

        elif edge_type == EDGETYPE_INTERFACE:
            edge_trace_list.append(trace)

        for i in [0, 1]:
            if edge[i] not in node_connect:
                node_connect[edge[i]] = 1
            else:
                node_connect[edge[i]] += 1
    node_trace_A = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(227,28,28)",
            size=[],
            line=dict(color=[], width=4, colorscale=cmap),
        ),
    )
    # 'rgb(227,28,28)'
    node_trace_B = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(0,102,255)",
            size=[],
            line=dict(color=[], width=4, colorscale=cmap),
        ),
    )
    # 'rgb(0,102,255)'
    node_trace = [node_trace_A, node_trace_B]

    for node in graph.nodes:

        if "chain" in graph.nodes[node]:
            index = int(graph.nodes[node]["chain"])
        else:
            index = 0

        pos = graph.nodes[node]["pos2D"]

        node_trace[index]["x"] += (pos[0],)
        node_trace[index]["y"] += (pos[1],)
        node_trace[index]["text"] += (
            "[Clst:" + str(cluster[node]) + "] " + " ".join(node),
        )

        nc = node_connect[node]
        node_trace[index]["marker"]["size"] += (5 + 15 * numpy.tanh(nc / 5),)
        node_trace[index]["marker"]["line"]["color"] += (cluster[node],)

    fig = go.Figure(
        data=[*internal_edge_trace_list, *edge_trace_list, *node_trace],
        layout=go.Layout(
            title="<br>tSNE connection graph",
            titlefont=dict(size=16),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if not disable_plot:
        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)


def plotly_3d( # pylint: disable=too-many-locals, too-many-branches
    graph: networkx.Graph,
    out: Optional[str] = None,
    offline: bool = False,
    iplot: bool = True,
    disable_plot: bool = False,
):
    """Plots interface graph in 3D

    Args:
        graph(deeprank graph object): the graph to be plotted
        out ([type], optional): [description]. Defaults to None.
        offline (bool, optional): [description]. Defaults to False.
        iplot (bool, optional): [description]. Defaults to True.
    """

    if offline:
        import plotly.offline as py # pylint: disable=import-outside-toplevel
    else:
        import chart_studio.plotly as py # pylint: disable=import-outside-toplevel

    edge_trace_list, internal_edge_trace_list = [], []
    node_connect = {}

    for edge in graph.edges:

        edge_type = _get_edge_type_name(
            graph.edges[edge[0], edge[1]][FEATURENAME_EDGETYPE]
        )
        if edge_type == EDGETYPE_INTERNAL:
            trace = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter3d.Line(color="rgb(110,110,110)", width=5),
            )

        elif edge_type == EDGETYPE_INTERFACE:
            trace = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                text=[],
                mode="lines",
                hoverinfo=None,
                showlegend=False,
                line=go.scatter3d.Line(color="rgb(210,210,210)", width=2),
            )
        else:
            continue

        x0, y0, z0 = graph.nodes[edge[0]][FEATURENAME_POSITION]
        x1, y1, z1 = graph.nodes[edge[1]][FEATURENAME_POSITION]

        trace["x"] += (x0, x1, None)
        trace["y"] += (y0, y1, None)
        trace["z"] += (z0, z1, None)

        if edge_type == EDGETYPE_INTERNAL:
            internal_edge_trace_list.append(trace)

        elif edge_type == EDGETYPE_INTERFACE:
            edge_trace_list.append(trace)

        for i in [0, 1]:
            if edge[i] not in node_connect:
                node_connect[edge[i]] = 1
            else:
                node_connect[edge[i]] += 1

    node_trace_A = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(227,28,28)",
            size=[],
            symbol="circle",
            line=dict(color="rgb(50,50,50)", width=2),
        ),
    )

    node_trace_B = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="rgb(0,102,255)",
            size=[],
            symbol="circle",
            line=dict(color="rgb(50,50,50)", width=2),
        ),
    )

    node_trace = [node_trace_A, node_trace_B]

    for node in graph.nodes:

        if FEATURENAME_CHAIN in graph.nodes[node]:
            index = int(graph.nodes[node][FEATURENAME_CHAIN])
        else:
            index = 0

        pos = graph.nodes[node]["pos"]

        node_trace[index]["x"] += (pos[0],)
        node_trace[index]["y"] += (pos[1],)
        node_trace[index]["z"] += (pos[2],)
        node_trace[index]["text"] += (" ".join(node),)

        nc = node_connect[node]
        node_trace[index]["marker"]["size"] += (5 + 15 * numpy.tanh(nc / 5),)

    fig = go.Figure(
        data=[*node_trace, *internal_edge_trace_list, *edge_trace_list],
        layout=go.Layout(
            title="<br>Connection graph",
            titlefont=dict(size=16),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if not disable_plot:
        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)
