import pickle
import os
from tqdm import tqdm

current_directory = os.getcwd()
parent_directory = os.path.split(current_directory)[0]
parent_directory = os.path.split(parent_directory)[0]
pickle_file_path = os.path.join(parent_directory, 'data', 'raw', 'first_million_articles.pkl')

with open(pickle_file_path, 'rb') as fp:
    articles = pickle.load(fp)


years = {}
articles_after_2008 = 0
articles_before_2008 = 0
for art in tqdm(articles):
    if art["date"][0] > 2006:
        articles_after_2008 += 1
    else:
        articles_before_2008 += 1
    if art["date"][0] in years:
        years[art["date"][0]] = years[art["date"][0]] + 1
    else:
        years[art["date"][0]] = 1

print("articles_before_2008:", articles_before_2008)
print("articles_after_2008:", articles_after_2008)
print("ratio:", articles_after_2008/articles_before_2008)