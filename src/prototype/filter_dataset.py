import pickle
import os
from tqdm import tqdm

current_directory = os.getcwd()
parent_directory = os.path.split(current_directory)[0]
parent_directory = os.path.split(parent_directory)[0]
pickle_file_path = os.path.join(parent_directory, 'data', 'raw', 'first_million_articles.pkl')

with open(pickle_file_path, 'rb') as fp:
    articles = pickle.load(fp)

mesh_terms = {}

for art in tqdm(articles):
    for mesh in art['mesh_names']:
        if mesh in mesh_terms:
            mesh_terms[mesh] = mesh_terms[mesh] + 1
        else:
            mesh_terms[mesh] = 1

reduced_set_of_mesh_terms = {}

for key,val in mesh_terms.items():
    if val > 100 and val < 50000:
        reduced_set_of_mesh_terms[key] = val


sorted_mesh_terms_by_popularity = sorted(reduced_set_of_mesh_terms.items(), key=lambda x:x[1])

print("number of mesh tags:", len(reduced_set_of_mesh_terms), "\n")

print("most common mesh tags:")
for i in range(10):
    print(sorted_mesh_terms_by_popularity[-(i + 1)])

print("\nleast common mesh tags:")
for i in range(10):
    print(sorted_mesh_terms_by_popularity[i])
