import numpy as np

def dict_of_c_given_a_weight(graph, a):
    b_nodes = list(graph.neighbors(a))
    c_nodes_count = {}
    for b in b_nodes:
        c_nodes = list(graph.neighbors(b))
        for c in c_nodes:
            if c not in b_nodes and c != a:
                if c in c_nodes_count:
                    c_nodes_count[c] += graph[b][c]["weight"] + graph[a][b]["weight"]
                else:
                    c_nodes_count[c] = graph[b][c]["weight"] + graph[a][b]["weight"]
    return c_nodes_count

def dict_of_c_given_a_frequency(graph, a):
    b_nodes = list(graph.neighbors(a))
    c_nodes_count = {}
    for b in b_nodes:
        c_nodes = list(graph.neighbors(b))
        for c in c_nodes:
            if c not in b_nodes and c != a:
                if c in c_nodes_count:
                    c_nodes_count[c] += 1
                else:
                    c_nodes_count[c] = 1
    return c_nodes_count

def dict_of_c_given_a_weight_frequency(graph, a):
    b_nodes = list(graph.neighbors(a))
    c_nodes_count = {}
    for b in b_nodes:
        c_nodes = list(graph.neighbors(b))
        for c in c_nodes:
            if c not in b_nodes and c != a:
                if c in c_nodes_count:
                    c_nodes_count[c]["weight"] += graph[b][c]["weight"] + graph[a][b]["weight"]
                    c_nodes_count[c]["frequency"] += 1
                else:
                    c_nodes_count[c] = {"weight": 0, "frequency": 1}
                    c_nodes_count[c]["weight"] = graph[b][c]["weight"] + graph[a][b]["weight"]
    for c in c_nodes_count:
        c_nodes_count[c] = np.sqrt(c_nodes_count[c]["weight"] * c_nodes_count[c]["frequency"])
    return c_nodes_count

def sort_c(c):
    sorted_c = sorted(c.items(), key=lambda x:x[1], reverse=True)
    return sorted_c