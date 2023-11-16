from tqdm import tqdm
import pickle
import dataset_functions
from itertools import combinations

path = dataset_functions.path_to_data(1, "first_million_articles.pkl")
before_2008_path = dataset_functions.path_to_data(1, "before_2008_pairs.pkl")
after_2008_path = dataset_functions.path_to_data(1, "after_2008_pairs.pkl")

with open(path, 'rb') as fp:
    articles = pickle.load(fp)

filtered_data = dataset_functions.filter_dataset_by_frequency(100, 50000, articles)

#dictionaries = {"tag1_tag2": value}

relations_before_2008 = {}
relations_after_2008 = {}

for article in tqdm(filtered_data):
    tags = article["mesh_names"]
    pairs = list(combinations(tags, 2))
    for pair in pairs:
        pair_name_0 = pair[0] + "_" + pair[1]
        pair_name_1 = pair[1] + "_" + pair[0]
        if article["date"][0] < 2008:
            if pair_name_0 in relations_before_2008:
                relations_before_2008[pair_name_0] += 1
            elif pair_name_1 in relations_before_2008:
                relations_before_2008[pair_name_1] += 1
            else:
                relations_before_2008[pair_name_0] = 1
        else:
            if pair_name_0 in relations_after_2008:
                relations_after_2008[pair_name_0] += 1
            elif pair_name_1 in relations_after_2008:
                relations_after_2008[pair_name_1] += 1
            else:
                relations_after_2008[pair_name_0] = 1

with open(before_2008_path, 'wb') as fp:
    pickle.dump(relations_before_2008, fp)

with open(after_2008_path, 'wb') as fp:
    pickle.dump(relations_after_2008, fp)

