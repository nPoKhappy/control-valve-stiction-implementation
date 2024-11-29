"""test combined data from data.py to see the R output"""
import numpy as np
from new_project.thesis_2.sigmoid_class import Sigmoid
import matplotlib.pyplot as plt
from new_project.final.function_file import aggregate_points

# Load final data
# Shape is (6625,)
combined_pv = np.loadtxt('new_project/csv/final/13_11_1_10_24_pv.csv', delimiter=',')
combined_op = np.loadtxt('new_project/csv/final/13_11_1_10_24_op.csv', delimiter=',')

print(f"combine_pv shape : {combined_pv.shape}\n")

# # 12_2_23_29 data issue (extract data only 11700 rows)
# combined_pv = combined_pv[:11700]
# combined_op = combined_op[:11700]

# Define the number of samples per sub-dataset
samples_per_set = 60 # 125 is the greatest common divisor # Of course, the number can be adjusted.


# Split combined_pv and combined_op into sub-datasets of 125 samples each
# pv and op split looks like [(), (), ()......()] list contain mutiple array
pv_split = np.array_split(combined_pv, np.arange(samples_per_set, len(combined_pv), samples_per_set))
op_split = np.array_split(combined_op, np.arange(samples_per_set, len(combined_op), samples_per_set))

pv_split = pv_split[:-1]
op_split = op_split[:-1]
# Count for which samples range we are dealing with
count = 0
stiction = []
for op, pv in zip(op_split, pv_split):
    pv = aggregate_points(pv)
    op = aggregate_points(op)
    # Instance simoid class
    sigmoid = Sigmoid(op, pv)
    # delta pv and op draw
    # sigmoid.delta_pv_op_plot(name= " ")
    # Stiction detect
    is_stiction, r_value, params = sigmoid.detect_stiction()
    stiction.append(r_value)
    
    # Add samples_per_set
    count += samples_per_set
    # print(f"Current dealing with {count-samples_per_set} to {count} samples.")
    
plt.figure(dpi=150)
plt.plot(stiction)
plt.title("Stiction through the time-series increase.")
plt.ylabel("R")
plt.xlabel("window")
plt.grid()
plt.show()
