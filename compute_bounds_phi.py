# imports
if (True):
    import json
    from argparse import ArgumentParser
    import networkx as nx
    import itertools
    from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from tqdm import tqdm
    import functools
    import ast
    #from sklearn.metrics.pairwise import cosine_similarity
    #from sentence_transformers import SentenceTransformer
    import numpy as np
    import matplotlib.pyplot as plt
    from queue import PriorityQueue
    #import tiktoken
    import math
    import numba
    from numba import cuda
    from multiprocessing import Process, Manager
    from sklearn.linear_model import LogisticRegression
    from IPython.display import display
    from brokenaxes import brokenaxes
    import re
    import matplotlib.ticker as ticker
    import statistics
    import copy
    import igraph as ig
    import leidenalg

    import os
    import sys
    from networkx.algorithms import community
    from networkx.algorithms.community import kernighan_lin_bisection
    import random
    from collections import Counter
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import pairwise_distances
    from scipy.sparse.csgraph import shortest_path
    import pickle
    from datetime import datetime
    import time
    import pyarrow.parquet as pq
    import fastparquet

    import concurrent.futures
    import time

    from pathlib import Path

    from class_pERbacco import *

    import binpacking
    

dict_min = dict()

for dname in ["cora", "funding", "voters", "wdc80", "camera", "synth_10000"]:
    if "synth" in dname:
        key_dname = dname
        dict_min[key_dname] = dict()
        synth_precision = "1.0"
    else:
        key_dname = dname
        dict_min[key_dname] = dict()
        synth_precision = False

    if "synth" not in dname:
        g, graph = read_graph(dname)
    else:
        g, graph = read_graph(dname, synth_precision)
    
    
    perbacco = class_entity(dname, graph, g, batch_size = 10, alg_community = "False", mu_benefit = "mean", lambda_w = "False")


    recall_count = 0
    for i,row in perbacco.df_ground_truth.iterrows():
        entity1,entity2 = row
        if perbacco.graph.has_edge(entity1,entity2):
            recall_count += 1

    dict_min[key_dname]["recall"] = recall_count/len(perbacco.df_ground_truth)
    precision = recall_count/len(perbacco.graph.edges())
    dict_min[key_dname]["precision"] = precision

    G_opt = nx.Graph()
    for idx, row in g.iterrows():
        u, v = row
        G_opt.add_edge(u, v, weight = 0, max_weight = 0)

    components = list(nx.connected_components(G_opt))

    # Sort by cardinality (size), descending
    components_sorted = sorted(components, key=len, reverse=True)
    list_size_comp = [len(x) for x in components_sorted]

    dict_min[dname+"_"+"Phi"] = dict()
    for batch_size in [2,5,10,20,40]:
        minimum_number_queries = 0
        for x in list_size_comp:
            minimum_number_queries += q_rec_k(x, batch_size)

        list_rest = []
        for x in list_size_comp:
            rest = r_rec_k(x, batch_size)
            if rest > 1:
                list_rest.append(rest)

        if list_rest:
            bin = binpacking.to_constant_volume(list_rest, batch_size)
        else:
            bin = list()

        upper_minimum_number_queries = minimum_number_queries + len(bin)
        lower_minimum_number_queries = minimum_number_queries + math.ceil(sum(list_rest)/batch_size)

        dict_min[dname+"_"+"Phi"][batch_size] = (lower_minimum_number_queries,upper_minimum_number_queries)


with open('results/Phi.json', 'w') as f:
    json.dump(dict_min, f)