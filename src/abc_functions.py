def dict_of_c_given_a_unweighted(graph, a):
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

def dict_of_c_given_a(graph, a):
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