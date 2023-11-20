from tqdm import tqdm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def create_lists(dictionary):

    head = []
    relation = []
    tail =[]

    for key, val in tqdm(dictionary.items()):
        pair = key.split("_")
        head.append(pair[0])
        tail.append(pair[1])
        relation.append(val)
    
    return head, relation, tail
    
def create_knowledge_graph(head, relation, tail):
    df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail})

    print("Dataframe complete")

    graph = nx.Graph()
    for _, row in tqdm(df.iterrows()):
        graph.add_edge(row['head'], row['tail'], label=row['relation'])

    print("Graph complete")
    
    return graph

def display_graph(graph, title):
    pos = nx.spring_layout(graph, seed=42, k=0.9)
    labels = nx.get_edge_attributes(graph, 'label')
    plt.figure(figsize=(12, 10))
    nx.draw(graph, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
    plt.title(title)
    plt.show()