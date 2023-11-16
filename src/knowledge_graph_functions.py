from tqdm import tqdm
import pandas as pd
import networkx as nx

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