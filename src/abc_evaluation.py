import dataset_functions as df
import abc_functions as af
import pickle

before_2008_subgraph_path = df.path_to_data(1, "before_2008_subgraph.pkl")
after_2008_subgraph_path = df.path_to_data(1, "after_2008_subgraph.pkl")

with open(before_2008_subgraph_path, 'rb') as fp:
    before_2008_graph = pickle.load(fp)
with open(after_2008_subgraph_path, 'rb') as fp:
    after_2008_graph = pickle.load(fp)
print("opened data")

count = 0
results = {}
for node in before_2008_graph.nodes():
    if count == 10:
        break
    if node in after_2008_graph:
        print(node)
        count += 1
        node_results = {"new": [], "not_new": []}
        get_c = af.dict_of_c_given_a_weight(before_2008_graph, node)
        c_sorted = af.sort_c(get_c)
        after_neighbours = list(after_2008_graph.neighbors(node))
        for i in range(10):
            name = c_sorted[i][0]
            if name not in after_neighbours:
                node_results["new"].append((name, i))
            else:
                node_results["not_new"].append((name, i))
        results[node] = node_results

total_new = 0
total_not_new = 0
for key,val in results.items():
    new = len(results[key]["new"])
    not_new = len(results[key]["not_new"])
    total_new += new
    total_not_new += not_new
    if not_new > 0:
        ratio = new/not_new
    else:
        ratio = new
    print(key, ": New =", new, ": Not new =", not_new, ": Number of new per not new =", ratio)
    if total_not_new > 0:
        total_ratio = total_new/total_not_new
    else:
        total_ratio = total_new
print("Total new =", total_new, ": Total not_new =", total_not_new, ": Total number of new per not new =", total_ratio)