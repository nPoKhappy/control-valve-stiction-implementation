import scipy.io
import numpy as np
from new_project.thesis_1.method1_class import Method1
from new_project.thesis_1.method2_class import Method2
from new_project.thesis_1.method3_class import Method3
from new_project.thesis_1.method4_class import Method4
from new_project.thesis_1.quantification_class import Quantification
from new_project.thesis_1.method_stiction_class import stiction_signal_plot
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
    
    """method1 stiction signal"""
    s_method1_loop2 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method1_loop2.csv', delimiter=',')
    S_method1_loop2_expand = np.repeat(s_method1_loop2, 8)
    """method2 stiction signal"""
    s_method2_loop2 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method2_loop2.csv', delimiter=',')
    S_method2_loop2_expand = np.repeat(s_method2_loop2, 8)
    """method3 stiction signal"""
    s_method3_loop2 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method3_loop2.csv', delimiter=',')
    S_method3_loop2_expand = np.repeat(s_method3_loop2, 8)
    """method4 stiction signal"""
    s_method4_loop2 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method4_loop2.csv', delimiter=',')
    S_method4_loop2_expand = np.repeat(s_method4_loop2, 3)

    """method1"""
    # method1 = Method1(pv, op)
    # print(f"the {loop_name} th loop")
    # method1.perform()
    # method1.store_csv("loop2")
    # print(f"sum of s:{sum(method1.s)}")
    """method2"""
    # method2 = Method2(pv, op)
    # method2.perform()
    # method2.store_csv("loop2")
    # print(f"sum of s:{sum(method2.s)}")
    """method3"""
    # method3 = Method3(pv, op)
    # method3.perform()
    # method3.store_csv("loop2")
    """method4"""
    method4 = Method4(pv, op)
    method4.perform()
    # method4.store_csv("loop2")
    # print(f"sum of s:{sum(method4.s)}")
    """quantification"""
    # quantification = Quantification(S_method4_loop2_expand, op_loop2)
    # quantification.perfrom()
    """stiction_signal on PV"""
    # stiction_signal_plot(pv, S_method4_loop2_expand)



