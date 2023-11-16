from tqdm import tqdm
import pickle
import dataset_functions as df
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

before_2008_path = df.path_to_data(1, "before_2008_pairs.pkl")
after_2008_path = df.path_to_data(1, "after_2008_pairs.pkl")

with open(before_2008_path, 'rb') as fp:
    before_2008_pairs = pickle.load(fp)

with open(after_2008_path, 'rb') as fp:
    after_2008_pairs = pickle.load(fp)

head = []
relation = []
tail =[]

i = 0
for key,val in tqdm(before_2008_pairs.items()):
    i += 1
    if i == 10:
        break
    pair = key.split("_")
    head.append(pair[0])
    tail.append(pair[1])
    relation.append(val)

df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail})

print("Dataframe complete")

graph = nx.Graph()
for _, row in tqdm(df.iterrows()):
    graph.add_edge(row['head'], row['tail'], label=row['relation'])

print("Graph complete")

pos = nx.spring_layout(graph, seed=42, k=0.9)
labels = nx.get_edge_attributes(graph, 'label')
plt.figure(figsize=(12, 10))
nx.draw(graph, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
plt.title('Knowledge Graph')
plt.show()