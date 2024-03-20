import dataset_functions as df
import pickle
import re

# =========================== OPEN DATA ===========================

'''
combination_results_1_path = df.path_to_data(1, "combination_results_1.pkl")
combination_results_2_path = df.path_to_data(1, "combination_results_2.pkl")
combination_results_3_path = df.path_to_data(1, "combination_results_3.pkl")
'''

txt_results_1_path = df.path_to_processed_data(1, "combination_results_1.txt")
txt_results_2_path = df.path_to_processed_data(1, "combination_results_2.txt")
txt_results_3_path = df.path_to_processed_data(1, "combination_results_3.txt")

'''
with open(combination_results_1_path, 'rb') as cr1:
    combination_results_1 = pickle.load(cr1)
with open(combination_results_2_path, 'rb') as cr2:
    combination_results_2 = pickle.load(cr2)
with open(combination_results_3_path, 'rb') as cr3:
    combination_results_3 = pickle.load(cr3)
'''

f1 = open(txt_results_1_path, "r")
f2 = open(txt_results_2_path, "r")
f3 = open(txt_results_3_path, "r")

print("opened data")

# =========================== CREATE NEW STRUCTURES ===========================

x_score_sum = {20:[0,0,0,0,0,0,0], 
          50:[0,0,0,0,0,0,0], 
          100:[0,0,0,0,0,0,0]}

x_common_sum = {20:[0,0,0,0,0,0,0], 
          50:[0,0,0,0,0,0,0], 
          100:[0,0,0,0,0,0,0]}

x_correct_sum = {20:[0,0,0,0,0,0,0], 
          50:[0,0,0,0,0,0,0], 
          100:[0,0,0,0,0,0,0]}

key_count = 18

# =========================== PROCESS PICKLED DATA ===========================
'''
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
'''

# =========================== PROCESS TEXT DATA ===========================

text_files = [f1,f2,f3]
for file in text_files:
    line_num = 0
    x = 0
    line = file.readline()
    while line != "":
        line_num += 1
        if (line_num < 20):
            line = file.readline()
            continue
        elif line[0] == "x":
            common = []
            correct = []
            score = []
            x = int(re.findall(r'[\d]+', line)[0])
            line = file.readline()
            line = file.readline()
            line = file.readline()
            common = re.findall(r'[\d]+', line)
            for i in range(7):
                x_common_sum[x][i] += int(common[i])
            line = file.readline()
            line = file.readline()
            line = file.readline()
            line = file.readline()
            correct = re.findall(r'[\d]+', line)
            for i in range(7):
                x_correct_sum[x][i] += int(correct[i])
            line = file.readline()
            line = file.readline()
            line = file.readline()
            line = file.readline()
            line = file.readline()
            score = re.findall(r'[\d]*[.][\d]+', line)
            for i in range(7):
                x_score_sum[x][i] += float(score[i])
        line = file.readline()
    file.close()

# =========================== PRINT ===========================

print("key_count:", key_count)
print("\nrelative score averages")
for x in x_score_sum:
    print(x, ":", [j/key_count for j in x_score_sum[x]])

print("\nrelative correct averages")
for x in x_correct_sum:
    print(x, ":", [j/key_count for j in x_correct_sum[x]])

print("\nrelative common averages")
for x in x_common_sum:
    print(x, ":", [j/key_count for j in x_common_sum[x]])