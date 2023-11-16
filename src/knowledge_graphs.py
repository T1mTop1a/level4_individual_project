from tqdm import tqdm
import pickle
import dataset_functions as df
import knowledge_graph_functions as kgf
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

before_2008_path = df.path_to_data(1, "before_2008_pairs.pkl")
after_2008_path = df.path_to_data(1, "after_2008_pairs.pkl")

with open(before_2008_path, 'rb') as fp:
    before_2008_pairs = pickle.load(fp)

with open(after_2008_path, 'rb') as fp:
    after_2008_pairs = pickle.load(fp)

before_2008_head, before_2008_relation, before_2008_tail = kgf.create_lists(before_2008_pairs)

after_2008_head, after_2008_relation, after_2008_tail = kgf.create_lists(after_2008_pairs)

before_2008_graph = kgf.create_knowledge_graph(before_2008_head, before_2008_relation, before_2008_tail)

after_2008_graph = kgf.create_knowledge_graph(after_2008_head, after_2008_relation, after_2008_tail)

# visualizing full graph. Running takes eternity
'''
pos = nx.spring_layout(graph, seed=42, k=0.9)
labels = nx.get_edge_attributes(graph, 'label')
plt.figure(figsize=(12, 10))
nx.draw(graph, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
plt.title('Knowledge Graph')
plt.show()
'''