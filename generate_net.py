import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from numpy.random import normal

from networkx.algorithms.community.quality import partition_quality


def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def plot_sbm(G, partition):
    """ Plots the Stochastic Block Model graph
    Parameters
    ----------
    G: graph
    seed: int
      random seed
    Returns
    -------
    None
    """
    pos = community_layout(G, partition)
    nx.draw(G, pos, node_color=list(partition.values()))
    plt.show()


def set_weight_edges(g, partition):

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            g[ni][nj]["weight"] = abs(normal(loc=0.0, scale=1.0))
        else:
            g[ni][nj]["weight"] = abs(normal(loc=1.0, scale=1.0))
    return g


def generate_sbm(size, probs, seed):
    """ Function to generate graph networks using 
    Stochastic Block Model
    Parameters
    ----------
    n: int
      Number of nodes
    r: float
      probability of the edge
    Returns
    -------
    random graph
    """
    return nx.stochastic_block_model(size, probs, seed=seed)


# Set size of each community
community_sizes = [10, 10, 10]
# Edge probabilities between each community
community_probs = [[0.9, 0.04, 0.04],
                   [0.04, 0.9, 0.04],
                   [0.04, 0.04, 0.7]]
seed = 42

G_sbm = generate_sbm(community_sizes, community_probs, seed)

# wt = [[normal, poisson],
#       [poisson, normal]]
# wtargs = [[dict(loc=3, scale=1), dict(lam=5)],
#           [dict(lam=5), dict(loc=3, scale=1)]]
# G_sbm = sbm(community_sizes, community_probs, wtargs=wtargs)
partition = community_louvain.best_partition(G_sbm, random_state=seed)
G_sbm = set_weight_edges(G_sbm, partition)

new_partition = [[k for (k, v) in partition.items() if v == i]
                 for i in range(max(partition.values())+1)]


print(partition)
print(new_partition)

coverage, performance = partition_quality(G_sbm, new_partition)

print(coverage, performance)

plot_sbm(G_sbm, partition)
