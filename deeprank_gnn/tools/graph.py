import logging
from typing import List, Optional
from copy import deepcopy

import numpy
from scipy.spatial import distance_matrix
import networkx
import community
import markov_clustering

from deeprank_gnn.models.graph import Graph, Node, Edge
from deeprank_gnn.models.structure import Atom, Residue
from deeprank_gnn.models.contact import AtomicContact, ResidueContact
from deeprank_gnn.domain.feature import FEATURENAME_POSITION, FEATURENAME_COVALENT
from deeprank_gnn.tools.embedding import manifold_embedding
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

            if numpy.any(numpy.isnan(feature_value)):
                _log.debug(f"edge {edge_key} {feature_name} has NaN")
                return True

    return False


def plotly_2d(graph: networkx.Graph,
              out: Optional[str] = None,
              offline: bool = False,
              iplot: bool = True,
              disable_plot: bool = False,
              method: str = 'louvain'):

    """Plots the interface graph in 2D

    Args:
        graph: the graph to plot
        out: output name. Defaults to None.
        offline: Defaults to False.
        iplot: Defaults to True.
        method: 'mcl' of 'louvain'. Defaults to 'louvain'.
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
        covalent = graph.edges[e][FEATURENAME_COVALENT]
        if not numpy.any(covalent):
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

        covalent = numpy.any(graph.edges[edge[0], edge[1]][FEATURENAME_COVALENT])
        if covalent:
            trace = go.Scatter(x=[], y=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                               line=go.scatter.Line(color='rgb(110,110,110)', width=3))
        else:
            trace = go.Scatter(x=[], y=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                               line=go.scatter.Line(color='rgb(210,210,210)', width=1))

        x0, y0 = graph.nodes[edge[0]]['pos2D']
        x1, y1 = graph.nodes[edge[1]]['pos2D']

        trace['x'] += (x0, x1, None)
        trace['y'] += (y0, y1, None)

        if covalent:
            internal_edge_trace_list.append(trace)

        else:
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

        covalent = numpy.any(graph.edges[edge[0], edge[1]][FEATURENAME_COVALENT])
        if covalent:
            trace = go.Scatter3d(x=[], y=[], z=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                 line=go.scatter3d.Line(color='rgb(110,110,110)', width=5))
        else:
            trace = go.Scatter3d(x=[], y=[], z=[], text=[], mode='lines', hoverinfo=None,  showlegend=False,
                                 line=go.scatter3d.Line(color='rgb(210,210,210)', width=2))

        x0, y0, z0 = graph.nodes[edge[0]]['pos']
        x1, y1, z1 = graph.nodes[edge[1]]['pos']

        trace['x'] += (x0, x1, None)
        trace['y'] += (y0, y1, None)
        trace['z'] += (z0, z1, None)

        if covalent:
            internal_edge_trace_list.append(trace)
        else:
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

def build_atomic_graph( # pylint: disable=too-many-locals
    atoms: List[Atom], graph_id: str, edge_distance_cutoff: float
) -> Graph:
    """Builds a graph, using the atoms as nodes.
    The edge distance cutoff is in Ångströms.
    """

    positions = numpy.empty((len(atoms), 3))
    for atom_index, atom in enumerate(atoms):
        positions[atom_index] = atom.position

    distances = distance_matrix(positions, positions, p=2)
    neighbours = distances < edge_distance_cutoff

    graph = Graph(graph_id)
    for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
        if atom1_index != atom2_index:

            atom1 = atoms[atom1_index]
            atom2 = atoms[atom2_index]
            contact = AtomicContact(atom1, atom2)

            node1 = Node(atom1)
            node2 = Node(atom2)
            node1.features[FEATURENAME_POSITION] = atom1.position
            node2.features[FEATURENAME_POSITION] = atom2.position

            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(Edge(contact))

    return graph


def build_residue_graph( # pylint: disable=too-many-locals
    residues: List[Residue], graph_id: str, edge_distance_cutoff: float
) -> Graph:
    """Builds a graph, using the residues as nodes.
    The edge distance cutoff is in Ångströms.
    It's the shortest interatomic distance between two residues.
    """

    # collect the set of atoms
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)

    positions = numpy.empty((len(atoms), 3))
    for atom_index, atom in enumerate(atoms):
        positions[atom_index] = atom.position

    distances = distance_matrix(positions, positions, p=2)
    neighbours = distances < edge_distance_cutoff

    graph = Graph(graph_id)
    for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
        if atom1_index != atom2_index:

            atom1 = atoms[atom1_index]
            atom2 = atoms[atom2_index]

            residue1 = atom1.residue
            residue2 = atom2.residue

            contact = ResidueContact(residue1, residue2)

            node1 = Node(residue1)
            node2 = Node(residue2)

            node1.features[FEATURENAME_POSITION] = numpy.mean(
                [atom.position for atom in residue1.atoms], axis=0
            )
            node2.features[FEATURENAME_POSITION] = numpy.mean(
                [atom.position for atom in residue2.atoms], axis=0
            )

            # The same residue will be added  multiple times as a node,
            # but the Graph class fixes this.
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(Edge(contact))

    return graph
