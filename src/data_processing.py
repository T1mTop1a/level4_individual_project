import dataset_functions as df
import pickle
import re

# =========================== OPEN DATA ===========================

combination_results_1_path = df.path_to_data(1, "combination_results_1.pkl")
combination_results_2_path = df.path_to_data(1, "combination_results_2.pkl")
combination_results_3_path = df.path_to_data(1, "combination_results_3.pkl")
combination_results_4_path = df.path_to_data(1, "combination_results_4.pkl")
combination_results_5_path = df.path_to_data(1, "combination_results_4.pkl")

txt_results_1_path = df.path_to_processed_data(1, "combination_results_1.txt")
txt_results_2_path = df.path_to_processed_data(1, "combination_results_2.txt")
txt_results_3_path = df.path_to_processed_data(1, "combination_results_3.txt")
txt_results_4_path = df.path_to_processed_data(1, "combination_results_4.txt")
txt_results_5_path = df.path_to_processed_data(1, "combination_results_4.txt")

with open(combination_results_1_path, 'rb') as cr1:
    combination_results_1 = pickle.load(cr1)
with open(combination_results_2_path, 'rb') as cr2:
    combination_results_2 = pickle.load(cr2)
with open(combination_results_3_path, 'rb') as cr3:
    combination_results_3 = pickle.load(cr3)
with open(combination_results_4_path, 'rb') as cr4:
    combination_results_4 = pickle.load(cr4)
with open(combination_results_5_path, 'rb') as cr5:
    combination_results_5 = pickle.load(cr5)

f1 = open(txt_results_1_path, "r")
f2 = open(txt_results_2_path, "r")
f3 = open(txt_results_3_path, "r")
f4 = open(txt_results_4_path, "r")
f5 = open(txt_results_5_path, "r")

print("opened data")

# =========================== CREATE NEW STRUCTURES ===========================

x_data_sum = {20:{"abc":0, "gnn":0, "mf":0, "abc x gnn":0, "gnn x mf":0, "mf x abc":0, "all":0}, 
          50:{"abc":0, "gnn":0, "mf":0, "abc x gnn":0, "gnn x mf":0, "mf x abc":0, "all":0}, 
          100:{"abc":0, "gnn":0, "mf":0, "abc x gnn":0, "gnn x mf":0, "mf x abc":0, "all":0}}

x_data_values = {20:{"abc x gnn":[], "gnn x mf":[], "mf x abc":[], "all":[]}, 
          50:{"abc x gnn":[], "gnn x mf":[], "mf x abc":[], "all":[]}, 
          100:{"abc x gnn":[], "gnn x mf":[], "mf x abc":[], "all":[]}}

key_count = len(combination_results_1) + len(combination_results_2) + len(combination_results_3) + len(combination_results_4) + len(combination_results_5)

# =========================== PROCESS PICKLED DATA ===========================

for key in combination_results_1:
    for combo in combination_results_1[key]:
        for x in combination_results_1[key][combo]:
            x_data_sum[x][combo] += combination_results_1[key][combo][x][1]
            x_data_values[x][combo].append(combination_results_1[key][combo][x][1])

for key in combination_results_2:
    for combo in combination_results_2[key]:
        for x in combination_results_2[key][combo]:
            x_data_sum[x][combo] += combination_results_2[key][combo][x][1]
            x_data_values[x][combo].append(combination_results_2[key][combo][x][1])

for key in combination_results_3:
    for combo in combination_results_3[key]:
        for x in combination_results_3[key][combo]:
            x_data_sum[x][combo] += combination_results_3[key][combo][x][1]
            x_data_values[x][combo].append(combination_results_3[key][combo][x][1])

for key in combination_results_4:
    for combo in combination_results_4[key]:
        for x in combination_results_4[key][combo]:
            x_data_sum[x][combo] += combination_results_4[key][combo][x][1]
            x_data_values[x][combo].append(combination_results_4[key][combo][x][1])

for key in combination_results_5:
    for combo in combination_results_5[key]:
        for x in combination_results_5[key][combo]:
            x_data_sum[x][combo] += combination_results_5[key][combo][x][1]
            x_data_values[x][combo].append(combination_results_5[key][combo][x][1])

# =========================== PROCESS TEXT DATA ===========================

text_files = [f1,f2,f3,f4,f5]
for file in text_files:
    x = 0
    line = file.readline()
    while line != "":
        if line[0] == "x":
            x = int(re.findall(r'[\d]+', line)[0])
        if line[0].isnumeric(): 
            scores = re.findall(r'[\d]*[.][\d]+', line)
            if len(scores) > 1:
                x_data_sum[x]["abc"] += float(scores[0])
                x_data_sum[x]["gnn"] += float(scores[1])
                x_data_sum[x]["mf"] += float(scores[2])
        line = file.readline()
    file.close()

# =========================== PRINT ===========================

print("key_count:", key_count)
print("\nrelative score averages")
for x in x_data_sum:
    for combo in x_data_sum[x]:
        print(x, combo, x_data_sum[x][combo]/key_count)