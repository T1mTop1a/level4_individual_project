import gzip
import os
import jsonlines
from tqdm import tqdm

def path_to_data(level_of_current_directory, data_filename):
    current_directory = os.getcwd()
    for i in range(level_of_current_directory):
        current_directory = os.path.split(current_directory)[0]
    file_path = os.path.join(current_directory, 'data', 'raw', data_filename)
    return file_path

def path_to_processed_data(level_of_current_directory, data_filename):
    current_directory = os.getcwd()
    for i in range(level_of_current_directory):
        current_directory = os.path.split(current_directory)[0]
    file_path = os.path.join(current_directory, 'data', 'processed', data_filename)
    return file_path

def get_x_articles_with_mesh_tag_names(file_path, x):
    with gzip.open(file_path,'rt') as f:
        reader = jsonlines.Reader(f)
        first_x_articles = []
        for i in tqdm(range(x)):
            article = reader.read()
            article_key_info = {}
            article_key_info['pmid'] = article['pmid']
            article_key_info['date'] = article['publication_date']
            mesh_names = []
            for mesh_tag in article['mesh']:
                mesh_names.append(mesh_tag['name'])
            article_key_info['mesh_names'] = mesh_names
            first_x_articles.append(article_key_info)
        return first_x_articles
    
def get_x_articles_withmesh_tags_names_qualifiers(file_path, x):
    with gzip.open(file_path,'rt') as f:
        reader = jsonlines.Reader(f)
        first_x_articles = []
        for i in tqdm(range(x)):
            article = reader.read()
            article_key_info = {}
            article_key_info['pmid'] = article['pmid']
            article_key_info['date'] = article['publication_date']
            mesh_names = []
            for mesh_tag in article['mesh']:
                if mesh_tag['qualifiers'].isEmpty:
                    mesh_names.append(mesh_tag['name'])
                else:
                    composite_name = mesh_tag['name']
                    for qualifier in mesh_tag['qualifiers']:
                        composite_name = composite_name + ' ' + qualifier['name']
                    mesh_names.append(composite_name)
            article_key_info['mesh_names'] = mesh_names
            first_x_articles.append(article_key_info)
        return first_x_articles
    
def filter_dataset_by_frequency(low, high, articles):
    mesh_terms_count = {}

    for article in tqdm(articles):
        for mesh in article['mesh_names']:
            if mesh in mesh_terms_count:
                mesh_terms_count[mesh] = mesh_terms_count[mesh] + 1
            else:
                mesh_terms_count[mesh] = 1

    filtered_articles = []

    for i in tqdm(range(len(articles))):
        filtered = {}
        filtered['pmid'] = articles[i]['pmid']
        filtered['date'] = articles[i]['date']
        filtered_mesh = []
        for tag in articles[i]['mesh_names']:
            if mesh_terms_count[tag] > low and mesh_terms_count[tag] < high:
                filtered_mesh.append(tag)
        filtered['mesh_names'] = filtered_mesh
        filtered_articles.append(filtered)
    
    return filtered_articles 

