import logging
from copy import deepcopy

import numpy
import networkx
import community
import markov_clustering

from deeprank_gnn.tools.embedding import manifold_embedding
from deeprank_gnn.domain.feature import FEATURENAME_EDGETYPE
from deeprank_gnn.domain.graph import EDGETYPE_INTERFACE, EDGETYPE_INTERNAL
from deeprank_gnn.models.graph import Graph


_log = logging.getLogger(__name__)


HDF5KEY_GRAPH_SCORE = "score"
HDF5KEY_GRAPH_NODENAMES = "nodes"
HDF5KEY_GRAPH_NODEFEATURES = "node_data"
HDF5KEY_GRAPH_EDGENAMES = "edges"
HDF5KEY_GRAPH_INTERNALEDGENAMES = "internal_edges"
HDF5KEY_GRAPH_EDGEINDICES = "edge_index"
HDF5KEY_GRAPH_INTERNALEDGEINDICES = "internal_edge_index"
HDF5KEY_GRAPH_EDGEFEATURES = "edge_data"
HDF5KEY_GRAPH_INTERNALEDGEFEATURES = "internal_edge_data"



def graph_has_nan(graph):
    for node_key, node_dict in graph.nodes.items():
        for feature_name, feature_value in node_dict.items():

            if numpy.any(numpy.isnan(feature_value)):
                _log.debug(f"node {node_key} {feature_name} has NaN")
                return True

    for edge_key, edge_dict in graph.edges.items():
        for feature_name, feature_value in edge_dict.items():

            if feature_name == FEATURENAME_EDGETYPE:
                continue  # is expected to be string

            if numpy.any(numpy.isnan(feature_value)):
                _log.debug(f"edge {edge_key} {feature_name} has NaN")
                return True

    return False


def graph_to_hdf5(graph, hdf5_file):
    """ Write a featured graph to an hdf5 file, according to deeprank standards.

        Args:
            graph (deeprank graph object): the input graph to write to the file
            hdf5_file (h5py file object): the output hdf5 file
    """

    graph_group = hdf5_file.create_group(graph.id)

    # store node names
    node_names = numpy.array([str(node) for node in graph.nodes]).astype('S')
    graph_group.create_dataset(HDF5KEY_GRAPH_NODENAMES, data=node_names)

    # store node features
    node_features_group = graph_group.create_group(HDF5KEY_GRAPH_NODEFEATURES)
    node_key_list = list(graph.nodes.keys())
    first_node_data = graph.nodes[node_key_list[0]]
    node_feature_names = list(first_node_data.keys())
    for node_feature_name in node_feature_names:

        node_feature_data = [node_data[node_feature_name] for node_key, node_data in graph.nodes.items()]

        node_features_group.create_dataset(node_feature_name, data=node_feature_data)

    # store edges
    edge_indices = []
    internal_edge_indices = []

    edge_names = []
    internal_edge_names = []

    first_edge_data = list(graph.edges.values())[0]
    edge_feature_names = list([name for name in first_edge_data.keys() if name != FEATURENAME_EDGETYPE])

    edge_feature_data = {name: [] for name in edge_feature_names}
    internal_edge_feature_data = {name: [] for name in edge_feature_names}

    for edge_key, edge_data in graph.edges.items():

        node1, node2 = edge_key
        edge_type = edge_data[FEATURENAME_EDGETYPE]

        node_index1 = node_key_list.index(node1)
        node_index2 = node_key_list.index(node2)

        if edge_type == EDGETYPE_INTERFACE:
            edge_indices.append((node_index1, node_index2))
            edge_names.append(str(edge_key))

        elif edge_type == EDGETYPE_INTERNAL:
            internal_edge_indices.append((node_index1, node_index2))
            internal_edge_names.append(str(edge_key))

        for edge_feature_name in edge_feature_names:
            if edge_type == EDGETYPE_INTERFACE:
                edge_feature_data[edge_feature_name].append(edge_data[edge_feature_name])

            elif edge_type == EDGETYPE_INTERNAL:
                internal_edge_feature_data[edge_feature_name].append(edge_data[edge_feature_name])

    graph_group.create_dataset(HDF5KEY_GRAPH_EDGENAMES, data=numpy.array(edge_names).astype('S'))
    graph_group.create_dataset(HDF5KEY_GRAPH_INTERNALEDGENAMES, data=numpy.array(internal_edge_names).astype('S'))

    graph_group.create_dataset(HDF5KEY_GRAPH_EDGEINDICES, data=edge_indices)
    graph_group.create_dataset(HDF5KEY_GRAPH_INTERNALEDGEINDICES, data=internal_edge_indices)

    edge_feature_group = graph_group.create_group(HDF5KEY_GRAPH_EDGEFEATURES)
    internal_edge_feature_group = graph_group.create_group(HDF5KEY_GRAPH_INTERNALEDGEFEATURES)
    for edge_feature_name in edge_feature_names:
        edge_feature_group.create_dataset(edge_feature_name, data=edge_feature_data[edge_feature_name])
        internal_edge_feature_group.create_dataset(edge_feature_name, data=internal_edge_feature_data[edge_feature_name])

    # store targets
    score_group = graph_group.create_group(HDF5KEY_GRAPH_SCORE)
    for target_name, target_value in graph.targets.items():
        score_group.create_dataset(target_name, data=target_value)

    return graph_group


def _get_node_key(value):
    if type(value) == str:
        return value

    key = ""
    for item in value:
        if type(item) == bytes or type(item) == numpy.bytes_:
            key = item.decode()

        elif type(item) == str:
            key += item

        else:
            key += str(item)

    return key


def hdf5_to_graph(graph_group):
    """ Read a hdf5 group back into a graph

        Args:
            graph_group(h5py group): the hdf5 group that was made from the graph

        Returns(deeprank graph object): the graph stored in the hdf5 group, node and edge keys will be strings
    """

    # read targets
    targets = {}
    if HDF5KEY_GRAPH_SCORE in graph_group:
        score_group = graph_group[HDF5KEY_GRAPH_SCORE]
        for target_name in score_group.keys():
            targets[target_name] = score_group[target_name][()]

    graph = Graph(graph_group.name, targets=targets)

    # read nodes
    node_names = [_get_node_key(key) for key in graph_group[HDF5KEY_GRAPH_NODENAMES][()]]
    node_features_group = graph_group[HDF5KEY_GRAPH_NODEFEATURES]
    node_features = {}
    node_feature_names = list(node_features_group.keys())
    for node_feature_name in node_feature_names:
        node_features[node_feature_name] = node_features_group[node_feature_name][()]

    for node_index, node_name in enumerate(node_names):
        graph.add_node(node_name)
        graph.nodes[node_name]
        for node_feature_name in node_feature_names:
            graph.nodes[node_name][node_feature_name] = node_features[node_feature_name][node_index]

    # read edges and internal edges
    for edge_type_name, edge_name_key, edge_index_key, edge_features_key in \
            [(EDGETYPE_INTERFACE, HDF5KEY_GRAPH_EDGENAMES, HDF5KEY_GRAPH_EDGEINDICES, HDF5KEY_GRAPH_EDGEFEATURES),
             (EDGETYPE_INTERNAL, HDF5KEY_GRAPH_INTERNALEDGENAMES, HDF5KEY_GRAPH_INTERNALEDGEINDICES, HDF5KEY_GRAPH_INTERNALEDGEFEATURES)]:

        edge_names = graph_group[edge_name_key][()]
        edge_node_indices = graph_group[edge_index_key][()]
        edge_features_group = graph_group[edge_features_key]
        edge_features = {}
        edge_feature_names = list(edge_features_group.keys())
        for edge_feature_name in edge_feature_names:
            edge_features[edge_feature_name] = edge_features_group[edge_feature_name][()]

        for edge_index, edge_name in enumerate(edge_names):
            node1_index, node2_index = edge_node_indices[edge_index]
            node1_name = node_names[node1_index]
            node2_name = node_names[node2_index]
            edge_key = (node1_name, node2_name)

            graph.add_edge(node1_name, node2_name)
            graph.edges[edge_key][FEATURENAME_EDGETYPE] = edge_type_name
            for edge_feature_name in edge_feature_names:
                graph.edges[edge_key][edge_feature_name] = edge_features[edge_feature_name][edge_index]

    return graph


def _get_edge_type_name(value):
    if type(value) == numpy.bytes_ or type(value) == bytes:

        return value.decode()
    else:
        return value


def plotly_2d(graph, out=None, offline=False, iplot=True,
              disable_plot=False, method='louvain'):

    """Plots the interface graph in 2D

    Args:
        graph(deeprank graph object): the graph to plot
        out ([type], optional): output name. Defaults to None.
        offline (bool, optional): Defaults to False.
        iplot (bool, optional): Defaults to True.
        method (str, optional): 'mcl' of 'louvain'. Defaults to 'louvain'.
    """

    if offline:
        import plotly.offline as py
    else:
        import chart_studio.plotly as py

    import plotly.graph_objs as go
    import matplotlib.pyplot as plt 

    pos = numpy.array(
        [v.tolist() for _, v in networkx.get_node_attributes(graph, 'pos').items()])
    pos2D = manifold_embedding(pos)
    dict_pos = {n: p for n, p in zip(graph.nodes, pos2D)}
    networkx.set_node_attributes(graph, dict_pos, 'pos2D')

    # remove interface edges for clustering
    gtmp = deepcopy(graph)
    ebunch = []
    for e in graph.edges:
        typ = graph.edges[e]['type']
        if isinstance(typ, bytes):
            typ = typ.decode('utf-8')
        if typ == EDGETYPE_INTERFACE:
            ebunch.append(e)
    gtmp.remove_edges_from(ebunch)

    if method == 'louvain':
        cluster = community.best_partition(gtmp)

    elif method == 'mcl':
        matrix = networkx.to_scipy_sparse_matrix(gtmp)
        # run MCL with default parameters
        result = markov_clustering.run_mcl(matrix)
        mcl_clust = markov_clustering.get_clusters(result)    # get clusters
        cluster = {}
        node_key = list(graph.nodes.keys())
        for ic, c in enumerate(mcl_clust):
            for node in c:
                cluster[node_key[node]] = ic

    # get the colormap for the clsuter line
    ncluster = numpy.max([v for _, v in cluster.items()])+1
    cmap = plt.cm.nipy_spectral
    N = cmap.N
    cmap = [cmap(i) for i in range(N)]
    cmap = cmap[::int(N/ncluster)]
    cmap = 'plasma'

    edge_trace_list, internal_edge_trace_list = [], []

    node_connect = {}
    for edge in graph.edges:

        edge_type = _get_edge_type_name(graph.edges[edge[0], edge[1]]['type'])
        if edge_type == EDGETYPE_INTERNAL:
            trace = go.Scatter(x=[], y=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                               line=go.scatter.Line(color='rgb(110,110,110)', width=3))

        elif edge_type == EDGETYPE_INTERFACE:
            trace = go.Scatter(x=[], y=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                               line=go.scatter.Line(color='rgb(210,210,210)', width=1))
        else:
            continue

        x0, y0 = graph.nodes[edge[0]]['pos2D']
        x1, y1 = graph.nodes[edge[1]]['pos2D']

        trace['x'] += (x0, x1, None)
        trace['y'] += (y0, y1, None)

        if edge_type == EDGETYPE_INTERNAL:
            internal_edge_trace_list.append(trace)

        elif edge_type == EDGETYPE_INTERFACE:
            edge_trace_list.append(trace)

        for i in [0, 1]:
            if edge[i] not in node_connect:
                node_connect[edge[i]] = 1
            else:
                node_connect[edge[i]] += 1
    node_trace_A = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
                              marker=dict(color='rgb(227,28,28)', size=[],
                                          line=dict(color=[], width=4, colorscale=cmap)))
    # 'rgb(227,28,28)'
    node_trace_B = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
                              marker=dict(color='rgb(0,102,255)', size=[],
                                          line=dict(color=[], width=4, colorscale=cmap)))
    # 'rgb(0,102,255)'
    node_trace = [node_trace_A, node_trace_B]

    for node in graph.nodes:

        if 'chain' in graph.nodes[node]:
            index = int(graph.nodes[node]['chain'])
        else:
            index = 0

        pos = graph.nodes[node]['pos2D']

        node_trace[index]['x'] += (pos[0],)
        node_trace[index]['y'] += (pos[1],)
        node_trace[index]['text'] += (
            '[Clst:' + str(cluster[node]) + '] ' + ' '.join(node),)

        nc = node_connect[node]
        node_trace[index]['marker']['size'] += (
            5 + 15*numpy.tanh(nc/5),)
        node_trace[index]['marker']['line']['color'] += (
            cluster[node],)

    fig = go.Figure(data=[*internal_edge_trace_list, *edge_trace_list, *node_trace],
                    layout=go.Layout(
        title='<br>tSNE connection graph for %s' % graph.id,
        titlefont=dict(size=16),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    if not disable_plot:
        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)

def plotly_3d(graph, out=None, offline=False, iplot=True, disable_plot=False):
    """Plots interface graph in 3D

    Args:
        graph(deeprank graph object): the graph to be plotted
        out ([type], optional): [description]. Defaults to None.
        offline (bool, optional): [description]. Defaults to False.
        iplot (bool, optional): [description]. Defaults to True.
    """

    if offline:
        import plotly.offline as py
    else:
        import chart_studio.plotly as py

    import plotly.graph_objs as go

    edge_trace_list, internal_edge_trace_list = [], []
    node_connect = {}

    for edge in graph.edges:

        edge_type = _get_edge_type_name(graph.edges[edge[0], edge[1]]['type'])
        if edge_type == EDGETYPE_INTERNAL:
            trace = go.Scatter3d(x=[], y=[], z=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                 line=go.scatter3d.Line(color='rgb(110,110,110)', width=5))

        elif edge_type == EDGETYPE_INTERFACE:
            trace = go.Scatter3d(x=[], y=[], z=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                 line=go.scatter3d.Line(color='rgb(210,210,210)', width=2))
        else:
            continue

        x0, y0, z0 = graph.nodes[edge[0]]['pos']
        x1, y1, z1 = graph.nodes[edge[1]]['pos']

        trace['x'] += (x0, x1, None)
        trace['y'] += (y0, y1, None)
        trace['z'] += (z0, z1, None)

        if edge_type == EDGETYPE_INTERNAL:
            internal_edge_trace_list.append(trace)

        elif edge_type == EDGETYPE_INTERFACE:
            edge_trace_list.append(trace)

        for i in [0, 1]:
            if edge[i] not in node_connect:
                node_connect[edge[i]] = 1
            else:
                node_connect[edge[i]] += 1

    node_trace_A = go.Scatter3d(x=[], y=[], z=[], text=[], mode='markers', hoverinfo='text',
                                marker=dict(color='rgb(227,28,28)', size=[], symbol='circle',
                                            line=dict(color='rgb(50,50,50)', width=2)))

    node_trace_B = go.Scatter3d(x=[], y=[], z=[], text=[], mode='markers', hoverinfo='text',
                                marker=dict(color='rgb(0,102,255)', size=[], symbol='circle',
                                            line=dict(color='rgb(50,50,50)', width=2)))

    node_trace = [node_trace_A, node_trace_B]

    for node in graph.nodes:

        if 'chain' in graph.nodes[node]:
            index = int(graph.nodes[node]['chain'])
        else:
            index = 0

        pos = graph.nodes[node]['pos']

        node_trace[index]['x'] += (pos[0],)
        node_trace[index]['y'] += (pos[1],)
        node_trace[index]['z'] += (pos[2], )
        node_trace[index]['text'] += (' '.join(node),)

        nc = node_connect[node]
        node_trace[index]['marker']['size'] += (5 + 15*numpy.tanh(nc/5), )

    fig = go.Figure(data=[*node_trace, *internal_edge_trace_list, *edge_trace_list],
                    layout=go.Layout(
                    title='<br>Connection graph for %s' % graph.id,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    if not disable_plot:
        if iplot:
            py.iplot(fig, filename=out)
        else:
            py.plot(fig)

