import gzip
import os
import jsonlines
from tqdm import tqdm
import pickle

current_directory = os.getcwd()
parent_directory = os.path.split(current_directory)[0]
parent_directory = os.path.split(parent_directory)[0]
file_path = os.path.join(parent_directory, 'data', 'raw', 'mesh_1410000.jsonl.gz')
pickle_file_path = os.path.join(parent_directory, 'data', 'raw', 'first_million_articles.pkl')

with gzip.open(file_path,'rt') as f:

    reader = jsonlines.Reader(f)
    first_1000000_articles = []

    article = reader.read()
    print(article)
    '''
    major_mesh = []
    for tag in article['mesh']:
        if tag['is_major'] == 'Y':
            major_mesh.append(tag['name'])

    print(major_mesh)

for i in tqdm(range(1000000)):
    article = reader.read()
    article_key_info = {}
    article_key_info["pmid"] = article["pmid"]
    article_key_info["date"] = article["publication_date"]
    mesh_names = []
    for mesh_tag in article["mesh"]:
        mesh_names.append(mesh_tag["name"])
    article_key_info["mesh_names"] = mesh_names
    first_1000000_articles.append(article_key_info)

with open(pickle_file_path, 'wb') as fp:
    pickle.dump(first_1000000_articles, fp)

'''