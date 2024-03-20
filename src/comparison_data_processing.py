import dataset_functions as df
import pickle
import re

# =========================== OPEN DATA ===========================

txt_results_1_path = df.path_to_processed_data(1, "comparison_results_1.txt")
txt_results_2_path = df.path_to_processed_data(1, "comparison_results_2.txt")
txt_results_3_path = df.path_to_processed_data(1, "comparison_results_3.txt")

f1 = open(txt_results_1_path, "r")
f2 = open(txt_results_2_path, "r")
f3 = open(txt_results_3_path, "r")

print("opened data")

# =========================== CREATE NEW STRUCTURES ===========================

x_data_sum = {1:{"abc":0, "gnn":0, "mf":0}, 
              10:{"abc":0, "gnn":0, "mf":0}, 
              20:{"abc":0, "gnn":0, "mf":0}, 
              50:{"abc":0, "gnn":0, "mf":0}, 
              100:{"abc":0, "gnn":0, "mf":0}}

x_data_count = {1:{"abc":0, "gnn":0, "mf":0}, 
              10:{"abc":0, "gnn":0, "mf":0}, 
              20:{"abc":0, "gnn":0, "mf":0}, 
              50:{"abc":0, "gnn":0, "mf":0}, 
              100:{"abc":0, "gnn":0, "mf":0}}

key_count = 18

# =========================== PROCESS TEXT DATA ===========================


text_files = [f1,f2,f3]
for file in text_files:
    x = 0
    line = file.readline()
    line_num = 0
    while line != "":
        line_num += 1
        if (line_num < 10):
            continue
        elif line[0] == "t": 
            line = file.readline()
            x = int(re.findall(r'[\d]+', line)[0])
            score = re.findall(r'[\d]*[.][\d]+', line)[0]
            x_data_sum[x]["abc"] += float(score)
            if float(score) > 0:
                x_data_count[x]["abc"] += 1

            line = file.readline()
            x = int(re.findall(r'[\d]+', line)[0])
            score = re.findall(r'[\d]*[.][\d]+', line)[0]
            x_data_sum[x]["gnn"] += float(score)
            if float(score) > 0:
                x_data_count[x]["gnn"] += 1

            line = file.readline()
            x = int(re.findall(r'[\d]+', line)[0])
            score = re.findall(r'[\d]*[.][\d]+', line)[0]
            x_data_sum[x]["mf"] += float(score)
            if float(score) > 0.0:
                x_data_count[x]["mf"] += 1

        line = file.readline()
    file.close()

# =========================== PRINT ===========================

print("\nrelative score averages")
for x in x_data_sum:
    for system in x_data_sum[x]:
        print(x, system, x_data_sum[x][system]/key_count)
    print("")

print("\nX counts")
for x in x_data_count:
    for system in x_data_count[x]:
        print(x, system, x_data_count[x][system])
    print("")