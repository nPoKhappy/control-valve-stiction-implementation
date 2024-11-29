import numpy as np
import scipy.io
from new_project.thesis_3.linear_regression_class import Linear_Rregression
from new_project.thesis_3.quantification_class import Quantificatioin
from new_project.thesis_3.method_stiction_class import stiction_signal_plot


"""import data from idsb"""
# Load the .mat file
mat_contents = scipy.io.loadmat('new_project/isdb10.mat')

# Access the 'cdata' structure within 'mat_contents'
cdata = mat_contents['cdata'][0,0]

# Access the 'chemicals' structure within 'cdata'
chemicals = cdata['chemicals'][0, 0]


loop_list = [1, 2, 3, 6, 10, 11, 12, 13, 14, 16, 23, 24, 29, 32]
loop_array = np.array(loop_list)



for loop_name in loop_array:

    # Access a specific loop
    loop = chemicals[f'loop{loop_name}'][0, 0]

    # Now you can access fields within 'loop2'
    pv = loop['PV'] # shape (samples num, 1)
    op = loop['OP'] # shape (samples num, 1)
    t = loop['t']

    """fit linear_regression model"""
    linear_regression = Linear_Rregression(pv=pv, op=op, t=t, draw_log_plot=1)
    linear_regression.detect_stiction()
    # linear_regression.store_csv("loop13")

    """stic_band quantification"""
    # quantificatioin = Quantificatioin(op=op, s=s_loop32)
    # quantificatioin.perform()


# s_loop32 = np.loadtxt('new_project/csv/essay_3/stiction_signal_linear_regression_loop32.csv', delimiter=',')
# s_loop32_expand =  np.repeat(s_loop32, 6)

# s_loop6 = np.loadtxt('new_project/csv/essay_3/stiction_signal_linear_regression_loop6.csv', delimiter=',')
# s_loop6_expand =  np.repeat(s_loop6, 6)

# s_loop2 = np.loadtxt('new_project/csv/essay_3/stiction_signal_linear_regression_loop2.csv', delimiter=',')
# s_loop2_expand =  np.repeat(s_loop2, 6)

# s_loop10 = np.loadtxt('new_project/csv/essay_3/stiction_signal_linear_regression_loop10.csv', delimiter=',')
# s_loop10_expand =  np.repeat(s_loop10, 6)

# s_loop13 = np.loadtxt('new_project/csv/essay_3/stiction_signal_linear_regression_loop13.csv', delimiter=',')
# s_loop13_expand =  np.repeat(s_loop13, 6)


"""stiction signal on PV"""
# stiction_signal_plot(pv=pv, S_expand=s_loop10_expand, process_name="loop10")

