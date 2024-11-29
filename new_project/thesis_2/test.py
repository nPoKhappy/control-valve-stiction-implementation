import scipy.io
from new_project.thesis_2.sigmoid_class import Sigmoid
import numpy as np
"""import data from idsb"""
# Load the .mat file
mat_contents = scipy.io.loadmat('new_project/isdb10.mat')

# Access the 'cdata' structure within 'mat_contents'
cdata = mat_contents['cdata'][0,0]

# Access the 'chemicals' structure within 'cdata'
chemicals = cdata['chemicals'][0, 0]
pulpPapers = cdata['pulpPapers'][0, 0]

loop_list = [1, 2, 3, 6, 10, 11, 12, 13, 23, 24, 29, 32]
loop_array = np.array(loop_list)

stiction_list = []

for loop_name in loop_array:
    # Access a specific loop
    loop = chemicals[f'loop{loop_name}'][0, 0]

    # Now you can access fields within 'loop2'
    pv = loop['PV'] # shape (1000, )
    op = loop['OP'] # shape (1000, )

    sigmoid = Sigmoid(co=op, pv=pv)
    # sigmoid.delta_pv_op_plot(name= loop_name)

    #Stiction detect
    is_stiction, r_value = sigmoid.detect_stiction()
    stiction_list.append(r_value)
    #Produce output
    if is_stiction:
        print(f"Control valve has stiction (R_{loop_name} = {r_value:.2f})")
    else:
        print(f"Control vable has no stiction (R_{loop_name} = {r_value:.2f})")
    # a, b (coefficient of sigmoid function)
    # print(f"fitting parameter a and b: s_a = {params[0]:.4f}")


# Convert list into array 
stiction_array = np.array(stiction_list)

#! not use right now
# Store into csv
# np.savetxt(f"new_project/csv/final/stiction_array.csv", stiction_array, delimiter=',', fmt="%.3f")

