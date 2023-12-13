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
results = {}
for node in before_2008_graph.nodes():
    if count == 100:
        break
    count += 1
    node_results = {"one": 0, "five": 0, "ten": 0}
    get_c = af.dict_of_c_given_a_weight(before_2008_graph, node)
    c_sorted = af.sort_c(get_c)
    after_neighbours = list(after_2008_graph.neighbors(node))
    c_count = 0
    for c in c_sorted:
        if c_count == 10:
            break
        if c[0] in after_neighbours:
            node_results["ten"] += 1
            if c_count == 0:
                node_results["one"] += 1
            if c_count < 5:
                node_results["five"] += 1
        c_count += 1
    results[node] = node_results

total_results = {"one": 0, "five": 0, "ten": 0}
for node in results:
    total_results["one"] += results[node]["one"]
    total_results["five"] += results[node]["five"]
    total_results["ten"] += results[node]["ten"]

print("one ratio =", total_results["one"]/100)
print("five ratio =", total_results["five"]/500)
print("ten ratio =", total_results["ten"]/1000)
