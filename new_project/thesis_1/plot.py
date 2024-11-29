import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from new_project.final.function_file import aggregate_points
# Load the .mat file
mat_contents = scipy.io.loadmat('new_project/isdb10.mat')

# Access the 'cdata' structure within 'mat_contents'
cdata = mat_contents['cdata'][0,0]

# Access the 'chemicals' structure within 'cdata'
chemicals = cdata['chemicals'][0, 0]


loop_num = 10

# sliced_sample_num = 500

# Access a specific loop
loop = chemicals[f'loop{loop_num}'][0, 0]

# Extracting the data
sp_loop = loop['SP']
pv_loop = loop['PV']
op_loop = loop['OP']

# sp_loop = sp_loop[:sliced_sample_num]
# pv_loop = pv_loop[:sliced_sample_num]
# op_loop = op_loop[:sliced_sample_num]

def plot_pv_sp_op( pv: np.ndarray, sp: np.ndarray, op: np.ndarray, loop_name : str) -> None:
    # Time axis (assuming time is in sequence and matches the length of your data)
    samples = range(len(pv))

    # Plotting
    plt.figure(figsize=(10, 8))

    # Subplot 1: Setpoint (SP) and Process Variable (PV)
    plt.subplot(2, 1, 1)
    plt.plot(samples, sp, label="Setpoint")
    plt.plot(samples, pv, label="Process variable", color='r')
    plt.xlabel("samples")
    plt.legend()
    plt.grid(True)
    plt.title("Closed loop signal of CHEM24")


    # # Subplot 2: Controller Output (OP)
    plt.subplot(2, 1, 2)
    plt.plot(samples, op, label="Controller output")
    plt.xlabel("samples")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Adding the overall figure title and other formatting
    # plt.suptitle(f"Closed loop signals of CHEM{loop_num}", fontsize=14)
    # plt.title("Closed loop signal of CHEM24")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save fig 
    # plt.savefig(f"test_svr_image/loop_{loop_name}.png", format = "png")
    # Display the plots
    plt.show()



# Test
if __name__ == "__main__":
    # Load final data
    # combined_pv = np.loadtxt('new_project/csv/final/modified data/scaled_13_11_1_10_24_pv.csv', delimiter=',')
    # combined_op = np.loadtxt('new_project/csv/final/modified data/scaled_13_11_1_10_24_op.csv', delimiter=',')
    # combined_pv = np.loadtxt('new_project/csv/final/scaled_12_2_23_29_pv.csv', delimiter=',')
    # combined_op = np.loadtxt('new_project/csv/final/scaled_12_2_23_29_op.csv', delimiter=',')
    # window_pv = combined_pv[100:160]
    # window_op = combined_op[100:160]
    # window_pv = aggregate_points(window_pv)
    # window_op = aggregate_points(window_op)
    plot_pv_sp_op(pv=pv_loop, sp=sp_loop, op=op_loop, loop_name=None)

