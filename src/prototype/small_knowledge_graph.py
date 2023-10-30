import pickle
import os
from tqdm import tqdm
import pandas as pd
import itertools 

current_directory = os.getcwd()
parent_directory = os.path.split(current_directory)[0]
parent_directory = os.path.split(parent_directory)[0]
pickle_file_path = os.path.join(parent_directory, 'data', 'raw', 'first_million_articles.pkl')

with open(pickle_file_path, 'rb') as fp:
    articles = pickle.load(fp)

head = []
relation = []
tail = []
for i in tqdm(range(10)):
    for mesh_tag in articles[i]["mesh_names"]:
        found = -1
        if mesh_tag in head:
            for t in 