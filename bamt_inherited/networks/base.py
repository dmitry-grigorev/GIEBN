from bamt.networks.base import BaseNetwork
from bamt.log import logger_network
from pyvis.network import Network

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from typing import Dict, Tuple, List, Callable, Optional, Type, Union, Any, Sequence

class BaseNetworkGI(BaseNetwork):
    """
    Minor redefinition of BAMT`s base network in how to plot it for gradation interaction
    :param outputdirectory: define a directory where to save output of plot
    :param random_state: fixate randomly generated pallet for gradations
    :param max_cat: how many gradations features have
    :param custom_mapper: defined how to name gradations of specific features
    """

    def __init__(self, outputdirectory: str, random_state=42, max_cat=3, custom_mapper: Dict[str, Dict[int, str]] = None):
        super().__init__()
        self.directory = outputdirectory
        self.random_state = random_state
        self.max_cat = max_cat
        self.custom_mapper = custom_mapper

    def plot(self, output: str):
        """
        Extended version of default plot for BAMT network which dyes nodes of the same level (e.g. Low, Mid or High)
        by the same color.
        output: str name of output file
        """
        if not output.endswith('.html'):
            logger_network.error("This version allows only html format.")
            return None

        G = nx.DiGraph()
        nodes = [node.name for node in self.nodes]
        G.add_nodes_from(nodes)
        G.add_edges_from(self.edges)

        network = Network(height="800px", width="100%", notebook=True, directed=nx.is_directed(G),
                          layout='hierarchical')
                            #, cdn_resources='in_line'
        nodes_sorted = np.array(
            list(nx.topological_generations(G)), dtype=object)

        # Qualitative class of colormaps
        q_classes = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',
                     'tab20b']

        hex_colors = []
        for cls in q_classes:
            rgb_colors = plt.get_cmap(cls).colors
            hex_colors.extend(
                [matplotlib.colors.rgb2hex(rgb_color) for rgb_color in rgb_colors])

        hex_colors = np.array(hex_colors)
        # Number_of_colors in matplotlib in Qualitative class = 144

        # nodes color pallete fixed
        np.random.seed(self.random_state)

        class_number = self.max_cat
        hex_colors_indexes = np.random.choice(class_number, class_number, replace=False)
        np.random.randint(0, class_number, class_number)

        if self.custom_mapper is not None:
            hex_colors_indexes = np.append(hex_colors_indexes,
                                           np.random.choice(list(range(class_number, len(hex_colors))),
                                                            len(self.custom_mapper.keys()), replace=False))

        hex_colors_picked = hex_colors[hex_colors_indexes]

        class2color = {cls: color for cls, color in zip(
            list(range(len(hex_colors_picked))), hex_colors_picked)}
        if self.custom_mapper is not None:
            customs = {k: v for k, v in
                       zip(self.custom_mapper.keys(), list(range(class_number, len(hex_colors_picked))))}
            name2class = {node.name:
                node.name.split("_")[-1] if "_".join(node.name.split("_")[:-1])
                                            not in self.custom_mapper.keys() else "_".join(node.name.split("_")[:-1])
                for node in self.nodes}
        else:
            name2class = {node.name: node.name.split("_")[-1] for node in self.nodes}

        if self.custom_mapper is not None:
            name_mapper = self.custom_mapper
        else:
            name_mapper = dict()
        if self.max_cat == 3:
            name_mapper['other'] = {0: 'Low', 1: 'Mid', 2: 'High'}
        else:
            name_mapper['other'] = {k: v for k, v in enumerate(list(range(self.max_cat)))}

        for level in range(len(nodes_sorted)):
            for node_i in range(len(nodes_sorted[level])):
                name = nodes_sorted[level][node_i]
                if self.custom_mapper is None or "_".join(name.split("_")[:-1]) not in self.custom_mapper.keys():
                    #print(name2class[name])
                    cls = int(name2class[name])
                else:
                    cls = customs[name2class[name]]
                    # name2class[customs[name[:(len(name)-1)]]]
                color = class2color[cls]
                if self.custom_mapper is None or "_".join(name.split("_")[:-1]) not in self.custom_mapper.keys():
                    network.add_node(name, label="_".join(name.split("_")[:-1]) + '_' + name_mapper['other'][cls], color=color,
                                     size=45, level=level, font={'size': 36},
                                     title=f"Узел байесовской сети {name} (Уровень {name_mapper['other'][cls]})")
                else:
                    network.add_node(name, label=name.split("_")[0] + '_' + self.custom_mapper[name2class[name]][
                        int(name[-1])], color=color, size=45,
                                     level=level, font={'size': 36},
                                     title=f"Узел байесовской сети {'_'.join(name.split('_')[:-1]) + '_' + self.custom_mapper[name2class[name]][int(name.split('_')[-1])]}")

        for edge in G.edges:
            network.add_edge(edge[0], edge[1])

        network.hrepulsion(node_distance=300, central_gravity=0.5)

        if not (os.path.exists(self.directory)):
            os.mkdir(self.directory)

        network.show_buttons(filter_=["physics"])
        return network.show(self.directory + '/' + output)
