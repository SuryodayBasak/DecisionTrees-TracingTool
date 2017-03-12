import csv
import numpy as np
import time

DATASET = []

with open('/home/suryo/MyStuff/MyDev/Stocks_ShuYash/examples/rf/training_set.csv', 'r') as csvfile:
	data = csv.reader(csvfile, delimiter=',')
	for row in data:
		temp = []

		for val in row:
			try:
				temp.append(float(val))

			except:
				if val == '':
					temp.append(None)

				else:
					temp.append(val)

		DATASET.append(temp)

data_matrix = np.array(DATASET)
data_matrix = data_matrix.transpose()
impurity_matrix = data_matrix

def ret_gini (part, current_feature_index):
	no_neg = 0.0
	no_pos = 0.0
	
	for row in part:
		if row[0] == -1:
			no_neg += 1
		else:
			no_pos += 1
			
	total = no_neg + no_pos
	print(total)
	return 1-((no_neg / total)**2+(no_pos / total)**2)

def trace_tree(data_rows, max_features, used_feature_indexes = []):
	
	max_gain = -999
	optimal_left = None
	optimal_right = None
	best_feature_index = None
	total_iterations = 0
	
	for current_feature_index in range(1, len(data_rows)):
		if current_feature_index not in used_feature_indexes:

			data_rows = np.array(data_rows)
			split_data = data_rows.transpose()
			split_data = np.array(split_data)
			print(split_data)
			split_data =  split_data[split_data[:,current_feature_index].argsort()]
			node_gini_impurity = ret_gini(split_data,current_feature_index)
			print("The node's impurity is: ", node_gini_impurity)
	
			for j in range(1, len(split_data)):
				left_part = split_data[:j]
				right_part = split_data[j:]
			
				gini_left = ret_gini(left_part,current_feature_index)
				gini_right = ret_gini(right_part,current_feature_index)
		
				N_left = float(len(left_part))
				N_right = float(len(right_part))
				total = N_left + N_right
				p_left = N_left/total
				p_right = N_right/total
		
				GiniGain = node_gini_impurity - (p_left*gini_left) - (p_right*gini_right)
				print(GiniGain)
				print(left_part)
				print(right_part)
				print('')
		
				if GiniGain > max_gain:
					max_gain = GiniGain
					optimal_left = left_part
					optimal_right = right_part
					best_feature_index = current_feature_index
				#time.sleep(1)
				total_iterations += 1

	if best_feature_index not in used_feature_indexes:
		used_feature_indexes.append(best_feature_index)
		
	print('Best gain is: ', max_gain)
	print('Best left branch: ')
	print(optimal_left)
	print('Best right branch: ')
	print(optimal_right)
	print(total_iterations)
	print(used_feature_indexes)

data_rows = []
data_rows.append(impurity_matrix[0])

data_rows.append(impurity_matrix[1])
data_rows.append(impurity_matrix[2])

print(len(data_rows))
trace_tree(data_rows, len(data_rows))

#TO DO: use a numbering for samples; within the function, perform a splitting of the dataset; recursive calls, etc.
