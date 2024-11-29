"""Combine vavle performance decay data."""
from new_project.thesis_1.plot import plot_pv_sp_op
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


# Load the .mat file
mat_contents = scipy.io.loadmat('new_project/isdb10.mat')

# Access the 'cdata' structure within 'mat_contents'
cdata = mat_contents['cdata'][0,0]

# Access the 'chemicals' structure within 'cdata'
chemicals = cdata['chemicals'][0, 0]
pulpPapers = cdata['pulpPapers'][0, 0]

loop_list = [1, 2, 3, 6, 10, 11, 12, 13, 23, 24, 29, 32]
loop_array = np.array(loop_list)
# Convert list into str like this 13_11_1_10_24
loop_str = "_".join(map(str, loop_list))

# Initialize lists to store pv and op arrays
pv_combined = []
op_combined = []
shape = []

for loop_name in loop_array:
    # Access a specific loop
    loop = chemicals[f'loop{loop_name}'][0, 0]
    # Now you can access fields within 'loop2'
    pv = loop['PV'].flatten() # shape (samples num, 1)
    op = loop['OP'].flatten() # shape (samples num, 1)
    """not use right now"""
    # Calculate delta pv and normalized it
    delta_pv = np.diff(pv, prepend=pv[0])
    mean_pv = np.mean(delta_pv)
    std_pv = np.std(delta_pv)
    pv = (delta_pv - mean_pv) / std_pv
    pv = np.reshape(pv, (-1, 1)) # reshape to (samples, 1)
    # Make op to [0, 1]
    op = (op - np.min(op)) / (np.max(op) - np.min(op)) 
    op = np.reshape(op, (-1, 1))
    # Append to the lists
    pv_combined.append(pv)
    op_combined.append(op)
    shape.append(pv.shape)

# Combine all pv and op along rows
pv_combined = np.vstack(pv_combined) # (6625, 1)
op_combined = np.vstack(op_combined) # (6625, 1)


"""wont use right now"""
def smooth_transition(signal1: np.ndarray, signal2: np.ndarray)->np.ndarray:
    transition_len = 700
    weights_signal1 = np.linspace(1, 0, transition_len)  # Gradually decreases from 1 to 0
    weights_signal2 = np.linspace(0, 1, transition_len)  # Gradually increases from 0 to 1
    # Extract the overlap regions
    overlap_signal1 = signal1[-transition_len:]
    overlap_signal2 = signal2[:transition_len]
    # Perform weighted averaging in the transition region
    smooth_transition = overlap_signal1 * weights_signal1 + overlap_signal2 * weights_signal2
    # Combine the signals
    combined_signal = np.concatenate((signal1[:-transition_len], smooth_transition, signal2[transition_len:]))
    combined_signal = np.reshape(combined_signal, (-1, 1))
    return  combined_signal # Shape would be (-1, 1) multiple rows one col

# loop13 = chemicals['loop13'][0, 0]
# loop11 = chemicals['loop11'][0, 0]
# op13 = loop13['OP'].flatten() 
# op11 = loop11['OP'].flatten() 
# pv13 = loop13['PV'].flatten() 
# pv11 = loop11['PV'].flatten() 

# print(f"shape of op : {op13.shape}\n")
# print(f"shape of pv : {pv13.shape}\n")


# # Make op to [0, 1]
# op13 = (op13 - np.min(op13)) / (np.max(op13) - np.min(op13)) 
# op11 = (op11 - np.min(op11)) / (np.max(op11) - np.min(op11)) 

# delta_pv13 = np.diff(pv13, prepend=pv13[0])
# mean_pv = np.mean(delta_pv13)
# std_pv = np.std(delta_pv13)
# pv13 = (delta_pv13 - mean_pv) / std_pv

# delta_pv11 = np.diff(pv11, prepend=pv11[0])
# mean_pv = np.mean(delta_pv11)
# std_pv = np.std(delta_pv11)
# pv11 = (delta_pv11 - mean_pv) / std_pv



# op_13_11 = smooth_transition(signal1=op13, signal2=op11)
# pv_13_11 = smooth_transition(signal1=pv13, signal2=pv11)

# print(f"shape of 13 11 combine : {op_13_11.shape}")

# Save data into csv file
np.savetxt(f"new_project/csv/final/modified data/scaled_13_11_pv.csv", pv_combined, delimiter=',')
np.savetxt(f"new_project/csv/final/modified data/scaled_13_11_op.csv", op_combined, delimiter=',')
print(f"shape of each database : {shape}\n")



