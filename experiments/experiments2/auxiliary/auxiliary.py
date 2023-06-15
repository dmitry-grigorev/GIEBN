import numpy as np

from bamt.log import logger_network
from pyvis.network import Network
from bamt.networks import discrete_bn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from itertools import product
from pgmpy.estimators import K2Score, BicScore, BDeuScore

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import matplotlib

import networkx as nx

import os

import random
import copy

plt.rcParams.update({'font.size': 14})

def print_histogram(feat, data, data_discretized_enc, encoder):
    """
    This functions plots histogram for feature feat and dyes each of its gradations (Low, Mid, High) in different colors
    """
    m = np.where(encoder.feature_names_in_ == feat)[0][0]
    bins = np.histogram_bin_edges(data[feat], bins=30)
    for k, i in enumerate(abs(encoder.bin_edges_[m][None, :] - bins[:, None]).argmin(axis=0)):
        bins[i] = encoder.bin_edges_[m][k]
    plt.figure(figsize=(12, 5))
    sns.histplot(x=data[feat], hue=0 * data_discretized_enc.iloc[:, 3 * m] +
                                   1 * data_discretized_enc.iloc[:, 3 * m + 1] +
                                   2 * data_discretized_enc.iloc[:, 3 * m + 2], bins=bins)
    plt.legend(labels=['High', 'Mid', 'Low'])
    for k in range(3 * m, 3 * m + 3):
        plt.axvline(x=data[data_discretized_enc.iloc[:, k] == 1][feat].mean(), color='red')


def create_blacklist(bn_nodes):
    """
    This function produces blacklist for current task: edges between gradations of one variable are not allowed
    """
    return [(node1, node2) for node1, node2 in product(bn_nodes, bn_nodes) if node1[:-1] == node2[:-1]]


def extract_categories(data):
    return [feat + str(int(k)) for feat in data.columns for k in
            sorted(data[feat].unique())]


def fix_categories(categories_to_fix, data, categories):
    data.rename(
        columns={(feat + str(categories_to_fix[feat] - 1)): (feat + str(categories_to_fix[feat])) for feat in
                 categories_to_fix.keys()}, inplace=True)
    for feat in categories_to_fix:
        categories.remove((feat + str(categories_to_fix[feat] - 1)))
        categories.append((feat + str(categories_to_fix[feat])))


def construct_by_quantiles(data: pd.DataFrame, categoricals: list, scoring, max_cat=3):
    data_discretized = data.copy(deep=True)
    conts = data.columns if categoricals is None else data.columns.difference(categoricals)
    for feat in conts:
        data_discretized[feat] = pd.qcut(data[feat], q=3, labels=False, duplicates='drop')

    categories = extract_categories(data_discretized)

    # categories which have the number of states less than it is required have to be fixed
    categories_to_fix = {feat: c for feat in conts if
                         1 < (c := len(data_discretized[feat].unique())) < max_cat}

    encoder = OneHotEncoder(sparse=False)
    data_discretized_enc = pd.DataFrame(encoder.fit_transform(X=data_discretized), columns=categories, dtype='uint8')

    fix_categories(categories_to_fix, data_discretized_enc, categories)

    ublacklist = create_blacklist(categories)

    params = {'bl_add': ublacklist}

    bn = learn_bn(data_discretized_enc, categories, params, scoring=scoring)

    return {'bn': bn,
            'encoder': encoder,
            'categories': categories,
            'disc_data': data_discretized_enc}


def construct_by_uniform(data: pd.DataFrame, categoricals: list, scoring, max_cat=3):
    data_discretized = data.copy(deep=True)
    conts = data.columns if categoricals is None else data.columns.difference(categoricals)
    for feat in conts:
        data_discretized[feat] = pd.cut(data[feat], bins=3, labels=False, duplicates='drop')

    categories = extract_categories(data_discretized)

    # categories which have the number of states less than it is required have to be fixed
    categories_to_fix = {feat: c for feat in conts if
                         1 < (c := len(data_discretized[feat].unique())) < max_cat}

    encoder = OneHotEncoder(sparse=False)
    data_discretized_enc = pd.DataFrame(encoder.fit_transform(X=data_discretized), columns=categories, dtype='uint8')

    fix_categories(categories_to_fix, data_discretized_enc, categories)

    ublacklist = create_blacklist(categories)

    params = {'bl_add': ublacklist}

    bn = learn_bn(data_discretized_enc, categories, params, scoring=scoring)

    return {'bn': bn,
            'encoder': encoder,
            'categories': categories,
            'disc_data': data_discretized_enc}


def construct_by_kmeans(data: pd.DataFrame, categoricals: list, scoring, max_cat=3):
    data_discretized = data.copy(deep=True)
    conts = data.columns if categoricals is None else data.columns.difference(categoricals)

    # categories which have the number of states less than it is required have to be fixed
    categories_to_fix = {feat: c for feat in conts if
                         1 < (c := len(data_discretized[feat].unique())) < max_cat}

    encoder = KBinsDiscretizer(strategy='kmeans', n_bins=3, random_state=42)
    data_discretized_enc = pd.DataFrame(
        encoder.fit_transform(X=data_discretized[data_discretized.columns.difference(categoricals)]).toarray(),
        dtype='uint8')

    categories = [feat + str(int(k)) for feat in encoder.feature_names_in_ for k in range(3)]

    data_discretized_enc.columns = categories

    fix_categories(categories_to_fix, data_discretized_enc, categories)

    for cat in categoricals:
        for k in data[cat].unique():
            data_discretized_enc[cat + str(k)] = (data[cat] == k).astype(int)
            categories.append(cat + str(k))

    ublacklist = create_blacklist(categories)

    params = {'bl_add': ublacklist}

    bn = learn_bn(data_discretized_enc, categories, params, scoring=scoring)

    return {'bn': bn,
            'encoder': encoder,
            'categories': categories,
            'disc_data': data_discretized_enc}


def learn_bn(data_discretized_enc, categories, params, scoring):
    all_edges = list()
    # Для демонстрации проблемы ансамблевое построение необязательно
    r = 1
    bn = discrete_bn.DiscreteBN()

    nodes_descriptor = {"types": {cat: 'disc' for _, cat in enumerate(categories)},
                        "signs": {}}
    bn.add_nodes(nodes_descriptor)

    for k in range(r):
        bn.add_edges(data_discretized_enc.astype("int32"), scoring_function=scoring, params=params,
                     progress_bar=False)
        all_edges += [tuple(e) for e in bn.edges.copy()]
        bn.edges = list()
        #print(f'{k + 1}/{r} BNs learnt in ensemble', end='\r')

    counter = Counter(all_edges)

    # voting construction of BN (if r == 1, there is no voting)
    bn.edges = [list(e) for e in list(counter) if counter[e] > (r // 2) or r == 1]
    return bn


def plot_cat(bn, directory: str, output: str, random_state=42, max_cat=3, custom_mapper=None):
    """
    Extended version of default plot for BAMT network which dyes nodes of the same level (Low, Mid or High) by the same color.
    output: str name of output file
    """
    if not output.endswith('.html'):
        logger_network.error("This version allows only html format.")
        return None

    G = nx.DiGraph()
    nodes = [node.name for node in bn.nodes]
    G.add_nodes_from(nodes)
    G.add_edges_from(bn.edges)

    network = Network(height="800px", width="100%", notebook=True, directed=nx.is_directed(G),
                      layout='hierarchical')

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

    np.random.seed(random_state)

    class_number = max_cat
    hex_colors_indexes = np.random.choice(class_number, class_number, replace=False)
    np.random.randint(0, class_number, class_number)
    # np.random.choice(class_number, max_cat, replace=False)
    if custom_mapper is not None:
        hex_colors_indexes = np.append(hex_colors_indexes,
                                       np.random.choice(list(range(class_number, len(hex_colors))),
                                                        len(custom_mapper.keys()), replace=False))

    hex_colors_picked = hex_colors[hex_colors_indexes]

    class2color = {cls: color for cls, color in zip(
        list(range(len(hex_colors_picked))), hex_colors_picked)}
    if custom_mapper is not None:
        customs = {k: v for k, v in zip(custom_mapper.keys(), list(range(class_number, len(hex_colors_picked))))}
        name2class = {node.name: (
            node.name[-1] if node.name[:(len(node.name) - 1)] not in custom_mapper.keys() else node.name[
                                                                                               :(len(node.name) - 1)])
            for node in bn.nodes}
    else:
        name2class = {node.name: node.name[-1] for node in bn.nodes}

    if custom_mapper is not None:
        name_mapper = custom_mapper
    else:
        name_mapper = dict()
    if max_cat == 3:
        name_mapper['other'] = {0: 'Low', 1: 'Mid', 2: 'High'}
    else:
        name_mapper['other'] = {k: v for k, v in enumerate(list(range(max_cat)))}

    for level in range(len(nodes_sorted)):
        for node_i in range(len(nodes_sorted[level])):
            name = nodes_sorted[level][node_i]
            if custom_mapper is None or name[:(len(name) - 1)] not in custom_mapper.keys():
                cls = int(name2class[name])
            else:
                cls = customs[name2class[name]]
                # name2class[customs[name[:(len(name)-1)]]]
            color = class2color[cls]
            if custom_mapper is None or name[:(len(name) - 1)] not in custom_mapper.keys():
                network.add_node(name, label=name[:(len(name) - 1)] + '_' + name_mapper['other'][cls], color=color,
                                 size=45, level=level, font={'size': 36},
                                 title=f"Узел байесовской сети {name} (Уровень {name_mapper['other'][cls]})")
            else:
                network.add_node(name, label=name[:(len(name) - 1)] + '_' + custom_mapper[name2class[name]][
                    int(name[-1])], color=color, size=45,
                                 level=level, font={'size': 36},
                                 title=f"Узел байесовской сети {name[:(len(name) - 1)] + '_' + custom_mapper[name2class[name]][int(name[-1])]}")

    for edge in G.edges:
        network.add_edge(edge[0], edge[1])

    network.hrepulsion(node_distance=300, central_gravity=0.5)

    if not (os.path.exists(directory)):
        os.mkdir(directory)

    network.show_buttons(filter_=["physics"])
    return network.show(directory + '/' + output)


def calculate_ratio(bn_edges, true_edges):
    return len([edge for edge in bn_edges if edge in true_edges]) / len(true_edges)


def calculate_reversed_ratio(bn_edges, true_edges):
    return len([edge for edge in bn_edges if edge not in true_edges and edge[::-1] in true_edges]) / len(
        true_edges)


def noising_standard(true_edges, bn_result, data, construct_func, force_dist):
    np.random.seed(42)

    for edge in true_edges:
        forced_edge = edge
        source_num = 0
        n_trials, counter = 20, 0
        state_index = bn_result['disc_data'][bn_result['disc_data'][forced_edge[1]] == 1].index
        m = state_index.shape[0]
        mean, std = data.loc[state_index, forced_edge[1][:-1]].mean(), data.loc[state_index, forced_edge[1][:-1]].std()

        for s in range(n_trials):
            data_kdisc = data.copy(deep=True)
            data_kdisc.loc[state_index, forced_edge[1][:-1]] += force_dist(mean, std, size=m)

            kbn1 = construct_func(data_kdisc, ['marker'])['bn']

            if forced_edge in kbn1.edges:
                counter += 1
            del kbn1
            del data_kdisc
        print(f'Noising of true edge {edge}`s in-node the edge occured in {counter / n_trials * 100}% of cases ')


def force_normal_noise(mean, std, size):
    return np.random.normal(mean, std, size=size)


def force_uniform_noise(mean, std, size):
    return np.random.uniform(-np.abs(mean) / 2, np.abs(mean) / 2, size=size)
