import gzip
import os
import jsonlines

current_directory = os.path.dirname(__file__)
parent_directory = os.path.split(current_directory)[0]
parent_directory = os.path.split(parent_directory)[0]
file_path = os.path.join(parent_directory, 'data', 'raw', 'mesh_1410000.jsonl.gz')

with gzip.open(file_path,'rt') as f:

    reader = jsonlines.Reader(f)

    documents = [ doc for doc in reader ]

len(documents)

 