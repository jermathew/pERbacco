# imports
if (True):
    import json
    from argparse import ArgumentParser
    import networkx as nx
    import itertools
    #from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
    import pandas as pd
    import numpy as np
    import torch
    #from torch.utils.data import Dataset
    from tqdm import tqdm
    import functools
    import ast
    import numpy as np
    import matplotlib.pyplot as plt
    from queue import PriorityQueue
    import math
    #import numba
    #from numba import cuda
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

# INPUT

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["cora", "camera", "funding", "voters", "cddb", "restaurant", "wdc20", "wdc50", "wdc80", "census", "synth_250", "synth_1000", "synth_5000", "synth_10000"])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--alg_community', type=str, choices=["louvain", "leiden", "lpa", "infomap", "False"])
    parser.add_argument('--lambda_w')
    parser.add_argument('--mu_benefit', type=str, choices=["brspecial", "brmean", "brmax"])

    # optimal
    parser.add_argument('--optimal', type=str)

    #synthetic
    parser.add_argument('--synth_precision')
    
    args = parser.parse_args()


    dname = args.dataset
    batch_size = args.batch_size
    alg_community = args.alg_community
    lambda_w = args.lambda_w
    if lambda_w != "False":
        lambda_w = float(lambda_w)
    mu_benefit = args.mu_benefit

    # optimal
    optimal = args.optimal



    #synthetic
    synth_precision = args.synth_precision
        
    if synth_precision != "False":
        synth_precision = float(synth_precision)
    

k_minimum_queries = 3


# COMPUTE max_query
if (True):
    # COMPUTE THE LOWER BOUND FOR THE MINIMUM NUMBER OF QUERIES
    g, graph = read_graph(dname, synth_precision)
  


    G_opt = nx.Graph()
    for idx, row in g.iterrows():
        u, v = row
        G_opt.add_edge(u, v, weight = 0, max_weight = 0)

    components = list(nx.connected_components(G_opt))

    # Sort by cardinality (size), descending
    components_sorted = sorted(components, key=len, reverse=True)
    list_size_comp = [len(x) for x in components_sorted]

    minimum_number_queries = 0
    for x in list_size_comp:
        minimum_number_queries += q_rec_k(x, batch_size)

    list_rest = []
    for x in list_size_comp:
        rest = r_rec_k(x, batch_size)
        if rest > 1:
            list_rest.append(rest)


    lower_minimum_number_queries = minimum_number_queries + math.ceil(sum(list_rest)/batch_size)
    #print("minimum_number_queries ", minimum_number_queries)

    max_query = lower_minimum_number_queries * k_minimum_queries




random.seed(42)
print("PRINT max_query for ", dname, " and batch_size", batch_size, max_query)



if (True):

    if (optimal == "True"):
        G_opt = nx.Graph()
        for idx, row in g.iterrows():
            u, v = row
            G_opt.add_edge(u, v, weight = 0, max_weight = 0)

        components = list(nx.connected_components(G_opt))

        # Sort by cardinality (size), descending
        components_sorted = sorted(components, key=len, reverse=True)

        count_component = 0
        constant = int(batch_size * (batch_size -1)/2 * len(components_sorted[0]))
        for component in components_sorted:
            count_component += 1
            for u, v in itertools.combinations(component, 2):
                G_opt[u][v]["max_weight"] = round(1 + 1/(count_component * constant),15)
                G_opt[u][v]["weight"] = round(1 + 1/(count_component * constant),15)

        alg_community = "False"
        lambda_w = "False"
        mu_benefit = "brmax"
        perbacco = class_entity(dname, G_opt, g, batch_size, alg_community, mu_benefit, lambda_w)
    else:
        #g, graph = read_graph(dname, synth_precision)
        perbacco = class_entity(dname, graph, g, batch_size, alg_community, mu_benefit, lambda_w)
    



    perbacco.create_list_community()
    results = pd.DataFrame(columns=["number_query", "kind", "temperature", "recall", "total_match", "len_df_benefit", "progressive_recall", "time"])
    number_query = 0
    list_temperature = list()
    list_match = list()
    list_match_community_batch = list()
    list_match_representative_batch = list()
    old_match = 0
    old_set_community = {0}
    list_change = list()
    perbacco.temperature = perbacco.batch_size
    recall = 0
    total_TIME = 0
    nodes = set(perbacco.graph.nodes())
    
    if len(perbacco.list_community) > 1:
        print("WITH COMMUNITY, record ratio is: ", perbacco.sum_heavy_comm/len(nodes))
        perbacco.with_community = "T"
    else:
        print("WO COMMUNITY")
        perbacco.list_community = [nodes]
        perbacco.create_dict_comm()
        perbacco.with_community = "F"


    start_TIME = time.perf_counter()
    first_part = 0
    TIME_query_start = time.perf_counter()

    
    if perbacco.with_community == "T":
        first_part = 1
        visited = set()
        for comm in perbacco.list_community[:-1]:
            if number_query <= max_query:
                visited = visited.union(set(comm))
                perbacco.query(comm, "skip")
                current = set(comm)

                while len(current) >= perbacco.batch_size:

                    H = perbacco.graph.subgraph(current).copy()
                    vertex_weight_sum = {node: sum(data['weight'] for _, _, data in H.edges(node, data=True)) for node in H.nodes()}
                    
                    # Sort all edges by weight
                    sorted_edges = sorted(H.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

                    dict_selected_vertices = {node: 0 for node in H.nodes()}

                    # compute the next batch via greedy_heaviest_subgraph and update list_batch and H 
                    query_comm = perbacco.greedy_heaviest_subgraph(H, vertex_weight_sum, sorted_edges, dict_selected_vertices)
                    current = current.difference(query_comm)

                    # do the query
                    number_query += 1
                    perbacco.query(query_comm, "entity")
                    TIME_query_stop = time.perf_counter()
                    TIME_query = TIME_query_stop - TIME_query_start
                    TIME_query_start = time.perf_counter()
                    recall, total_match = perbacco.compute_recall()
                    list_temperature.append(round(perbacco.temperature,2))
                    list_match.append(recall)

                    print(number_query,  dname, batch_size, "COMMUNITY", "temperature: ", round(perbacco.temperature,2), "recall: ", round(recall,3), "total match: ", total_match, "len(benefit): ", len(perbacco.df_benefit), "delta match", total_match-old_match, query_comm, TIME_query)
                    results.loc[len(results)] = [number_query, "COMMUNITY", round(perbacco.temperature,2), round(recall,3), total_match, len(perbacco.df_benefit), total_match-old_match, TIME_query]
                    list_match_community_batch.append(total_match-old_match)
                    old_match = total_match

                    set_higher_temperature  = perbacco.compute_entity_higher_temperature()
                    while len(set_higher_temperature) == perbacco.batch_size:
                        current = current.difference(set(set_higher_temperature))

                        number_query += 1

                        # do the query and update the df_benefit
                        perbacco.query(set_higher_temperature, "entity")
                        TIME_query_stop = time.perf_counter()
                        TIME_query = TIME_query_stop - TIME_query_start
                        TIME_query_start = time.perf_counter()

                        recall, total_match = perbacco.compute_recall()
                        list_match.append(recall)

                        print(number_query, dname, batch_size, "CURRENT", "temperature: ", round(perbacco.temperature,2), "recall: ", round(recall,3), "total match: ", total_match, "len(benefit): ", len(perbacco.df_benefit), "delta match", total_match-old_match, set_higher_temperature, TIME_query)
                        results.loc[len(results)] = [number_query, "CURRENT", round(perbacco.temperature,2), round(recall,3), total_match, len(perbacco.df_benefit), (total_match-old_match), TIME_query]

                        list_match_representative_batch.append(total_match-old_match)
                        if len(list_match_community_batch) > 0:
                            threshold_temp = statistics.mean(list_match_community_batch)
                            
                        else:
                            threshold_temp = perbacco.batch_size / 2

                        if list_match_representative_batch[-1] <= threshold_temp:
                            perbacco.temperature *=2
                            list_temperature.append(round(perbacco.temperature,2))
                        else:
                            list_temperature.append(round(perbacco.temperature,2))


                        # compute again the set_higher_temperature
                        set_higher_temperature = perbacco.compute_entity_higher_temperature()

                        old_match = total_match

                    # decrease the temperature
                    perbacco.temperature *= (1-1/perbacco.batch_size)
    
    number_query_first_part = number_query
    stop_TIME = time.perf_counter()
    total_TIME += stop_TIME - start_TIME
    #if (optimal != "True"):
    #    if first_part:
    #        first_time = (stop_TIME - start_TIME)/ number_query
    #        with open("time.txt", "a") as f:
    #            f.write(f"{dname}_{batch_size}: pErbacco time per query first part = {first_time:.3f}, partial time = {total_TIME:.3f}, partial queries = {int(number_query)}\n")


    if number_query <= max_query:
        # LAST COMMUNITY
        all_nodes = list(perbacco.list_community[-1])
        perbacco.query(all_nodes, "last")

    
    
    perbacco.temperature = 0  
    start_TIME = time.perf_counter()
    second_part = 0
    random.seed(42)
    perbacco.df_benefit = perbacco.df_benefit.sort_values(by = "benefit", ascending=False, ignore_index=False) 


    while number_query <= max_query and len(perbacco.df_benefit) > 0:
        second_part = 1
        number_query += 1
        batch = perbacco.compute_entity_higher_temperature()

        perbacco.query(batch, "entity")
        TIME_query_stop = time.perf_counter()
        TIME_query = TIME_query_stop - TIME_query_start
        TIME_query_start = time.perf_counter()
        
        recall, total_match = perbacco.compute_recall()

        list_temperature.append(0)
        list_match.append(recall)
        print(number_query,  dname, batch_size, "CURRENT", "temperature: ", round(perbacco.temperature,2), "recall: ", round(recall,3), "total match: ", total_match, "len(benefit): ", len(perbacco.df_benefit), "delta match", total_match-old_match, batch,TIME_query)
        results.loc[len(results)] = [number_query, "CURRENT", round(perbacco.temperature,2), round(recall,3), total_match, len(perbacco.df_benefit), total_match-old_match,TIME_query]
        old_match = total_match
        list_match_community_batch.append(total_match-old_match)


    stop_TIME = time.perf_counter()
    partial_TIME = stop_TIME - start_TIME
    total_TIME += stop_TIME - start_TIME
    if (optimal != "True"):
        if second_part:
            time_per_query = total_TIME / number_query

            if alg_community == "False":
                if perbacco.mu_benefit == "brmax":
                    with open("time.txt", "a") as f:
                        f.write(f"{dname}_{batch_size} Online: time per query = {time_per_query:.3f}, total time = {total_TIME:.3f}, total queries = {int(number_query)}\n")
                else:
                    with open("time.txt", "a") as f:
                        f.write(f"{dname}_{batch_size} pERbac: time per query = {time_per_query:.3f}, total time = {total_TIME:.3f}, total queries = {int(number_query)}\n")
            else:
                
                partial_time_per_query = partial_TIME / (number_query - number_query_first_part)
                with open("time.txt", "a") as f:
                    f.write(f"{dname}_{batch_size} pERbacco: time per query second part = {partial_time_per_query:.3f}, partial time = {partial_TIME:.3f}, partial queries = {int(number_query - number_query_first_part)}\n")
                    
                    
                    f.write(f"{dname}_{batch_size} pERbacco: time per query = {time_per_query:.3f}, total time = {total_TIME:.3f}, total queries = {int(number_query)}\n")

























































list_size = list()
list_match = list()
total_size = 0
total_match = 0

if len(perbacco.list_community) > 1:
    for i in range(len(perbacco.list_community)):
        size, match, match_star = perbacco.info_plot_community(perbacco.list_community[i], plot = False)
        total_size += size
        total_match += match
        list_match.append(total_match)
        list_size.append(total_size)


prefix_lists =  str(batch_size)+","+str(alg_community)+","+str(lambda_w)


lists = {
    "probmatch" : perbacco.list_prob_match,
    "probnomatch" : perbacco.list_prob_no_match,
    "benefitmatch" : perbacco.list_benefit_match,
    "benefitnomatch" : perbacco.list_benefit_no_match,
    "listmatch" : list_match,
    "listsize" : list_size,
    "sum_heavy" : perbacco.sum_heavy_comm,
    "number_heavy" : perbacco.number_heavy,
}

directory = "results/lists_match/" + dname
if not os.path.exists(directory):
    os.makedirs(directory)

with open(directory + "/" + prefix_lists + ".json", 'w') as f:
    json.dump(lists, f)


directory = "results/"+dname
if not os.path.exists(directory):
    os.makedirs(directory)




if optimal == "False":
    if "synth" in dname:
        results.to_csv(directory+"/"+ dname+","+str(synth_precision)+","+str(batch_size)+","+str(alg_community[:3])+","+str(perbacco.mu_benefit)+".csv", index=False)
    else:
        if alg_community != "False":
            results.to_csv(directory+"/"+ dname+"_pERbacco,"+str(batch_size)+","+str(alg_community[:3])+","+str(lambda_w)+".csv", index=False) 
        else:
            if perbacco.mu_benefit == "brmean":
                results.to_csv(directory+"/"+ dname+"_Online,"+str(batch_size)+".csv", index=False) 
            else:
                results.to_csv(directory+"/"+ dname+"_pERbac,"+str(batch_size)+".csv", index=False) 

else:
    results.to_csv(directory+"/"+ dname+"_"+"suboptimal"+","+str(batch_size)+".csv", index=False) 

