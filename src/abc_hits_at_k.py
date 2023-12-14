# hits @k thoughts
# want to evaluate over a selection of nodes
# calculate for a variety of k. Maybe 1, 5 and 10
# The more hits the more valuable the misses are

import dataset_functions as df
import abc_functions as af
import pickle

before_2008_subgraph_path = df.path_to_data(1, "before_2008_subgraph.pkl")
after_2008_subgraph_path = df.path_to_data(1, "after_2008_subgraph.pkl")

with open(before_2008_subgraph_path, 'rb') as bg:
    before_2008_graph = pickle.load(bg)
with open(after_2008_subgraph_path, 'rb') as ag:
    after_2008_graph = pickle.load(ag)
print("opened data")

count = 0
results_weight = {}
results_frequency = {}
results_weight_frequency = {}
for node in before_2008_graph.nodes():
    if count == 100:
        break
    count += 1

    node_results_weight = {"one": 0, "five": 0, "ten": 0}
    get_c_weight = af.dict_of_c_given_a_weight(before_2008_graph, node)
    c_sorted_weight = af.sort_c(get_c_weight)
    after_neighbours_weight = list(after_2008_graph.neighbors(node))
    c_count = 0
    for c in c_sorted_weight:
        if c_count == 10:
            break
        if c[0] in after_neighbours_weight:
            node_results_weight["ten"] += 1
            if c_count == 0:
                node_results_weight["one"] += 1
            if c_count < 5:
                node_results_weight["five"] += 1
        c_count += 1
    results_weight[node] = node_results_weight

    node_results_frequency = {"one": 0, "five": 0, "ten": 0}
    get_c_frequency = af.dict_of_c_given_a_frequency(before_2008_graph, node)
    c_sorted_frequency = af.sort_c(get_c_frequency)
    after_neighbours_frequency = list(after_2008_graph.neighbors(node))
    c_count = 0
    for c in c_sorted_frequency:
        if c_count == 10:
            break
        if c[0] in after_neighbours_frequency:
            node_results_frequency["ten"] += 1
            if c_count == 0:
                node_results_frequency["one"] += 1
            if c_count < 5:
                node_results_frequency["five"] += 1
        c_count += 1
    results_frequency[node] = node_results_frequency

    node_results_weight_frequency = {"one": 0, "five": 0, "ten": 0}
    get_c_weight_frequency = af.dict_of_c_given_a_weight_frequency(before_2008_graph, node)
    c_sorted_weight_frequency = af.sort_c(get_c_weight_frequency)
    after_neighbours_weight_frequency = list(after_2008_graph.neighbors(node))
    c_count = 0
    for c in c_sorted_weight_frequency:
        if c_count == 10:
            break
        if c[0] in after_neighbours_weight_frequency:
            node_results_weight_frequency["ten"] += 1
            if c_count == 0:
                node_results_weight_frequency["one"] += 1
            if c_count < 5:
                node_results_weight_frequency["five"] += 1
        c_count += 1
    results_weight_frequency[node] = node_results_weight_frequency

total_results_weight = {"one": 0, "five": 0, "ten": 0}
for node in results_weight:
    total_results_weight["one"] += results_weight[node]["one"]
    total_results_weight["five"] += results_weight[node]["five"]
    total_results_weight["ten"] += results_weight[node]["ten"]

total_results_frequency = {"one": 0, "five": 0, "ten": 0}
for node in results_frequency:
    total_results_frequency["one"] += results_frequency[node]["one"]
    total_results_frequency["five"] += results_frequency[node]["five"]
    total_results_frequency["ten"] += results_frequency[node]["ten"]

total_results_weight_frequency = {"one": 0, "five": 0, "ten": 0}
for node in results_weight_frequency:
    total_results_weight_frequency["one"] += results_weight_frequency[node]["one"]
    total_results_weight_frequency["five"] += results_weight_frequency[node]["five"]
    total_results_weight_frequency["ten"] += results_weight_frequency[node]["ten"]

print("Results:")
print("")
print("Weight only:")
print("one ratio =", total_results_weight["one"]/100)
print("five ratio =", total_results_weight["five"]/500)
print("ten ratio =", total_results_weight["ten"]/1000)
print("")
print("Frequency only:")
print("one ratio =", total_results_frequency["one"]/100)
print("five ratio =", total_results_frequency["five"]/500)
print("ten ratio =", total_results_frequency["ten"]/1000)
print("")
print("Weight and Frequency:")
print("one ratio =", total_results_weight_frequency["one"]/100)
print("five ratio =", total_results_weight_frequency["five"]/500)
print("ten ratio =", total_results_weight_frequency["ten"]/1000)
print("")