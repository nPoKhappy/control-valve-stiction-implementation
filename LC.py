"""
Liquid Level Control
FT_115: PV
LV_106: CO 
"""

import pandas as pd
import numpy as np
# Thesis 1
from new_project.thesis_1.method1_class import Method1
from new_project.thesis_1.method2_class import Method2
from new_project.thesis_1.method3_class import Method3
from new_project.thesis_1.method4_class import Method4
from new_project.thesis_1.quantification_class import Quantification
from new_project.thesis_1.method_stiction_class import stiction_signal_plot
from new_project.thesis_1.plot import plot_pv_sp_op
# Thesis 2
from new_project.thesis_2.sigmoid_class import Sigmoid
# Thesis 3
from new_project.thesis_3.linear_regression_class import Linear_Rregression
from new_project.thesis_3.quantification_class import Quantificatioin
from new_project.thesis_3.method_stiction_class import stiction_signal_plot

# Load data 
# Load the CSV file
file_path = "new_project/LIC-106_20241121~27.csv"  
df = pd.read_csv(file_path)

# Convert each column directly into arrays
pv = df["FT_115"].to_numpy()
co = df["LV_106"].to_numpy()
# Reverse array since time is reversed
pv = np.flip(pv)
co = np.flip(co)

# Plot PV SP, and CO
plot_pv_sp_op(pv = pv, sp = None,co = co, loop_name = "liquid")

# Thesis 1 (Statistical)
"""# Method1"""
# method1 = Method1(pv, co)
# method1.perform()
# method1.store_csv("liquid")
# print(f"sum of s:{sum(method1.s)}")

"""# Method2"""
# method2 = Method2(pv, co)
# method2.perform()
# method2.store_csv("liquid")
# print(f"sum of s:{sum(method2.s)}")
"""# Method3"""
# method3 = Method3(pv, co)
# method3.perform()
# method3.store_csv("liquid")
"""# Method4"""
# method4 = Method4(pv, co)
# method4.perform()
# method4.store_csv("liquid")

"""Use after the csv stored"""
"""method1 stiction signal"""
# s_method1 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method1_liquid.csv', delimiter=',')
# S_method1_expand = np.repeat(s_method1, 8)
"""method2 stiction signal"""
# s_method2 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method2_liquid.csv', delimiter=',')
# S_method2_expand = np.repeat(s_method2, 8)
"""method3 stiction signal"""
# s_method3 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method3_liquid.csv', delimiter=',')
# S_method3_expand = np.repeat(s_method3, 8)
"""method4 stiction signal"""
# s_method4 = np.loadtxt('new_project/csv/essay_1/stiction_signal_method4_liquid.csv', delimiter=',')
# S_method4_expand = np.repeat(s_method4, 3)


"""quantification"""
# quantification = Quantification(S_method2_expand, co) # Change the former parameter name (method 1 ~ 4)
# quantification.perfrom()

"""stiction_signal on PV"""
# stiction_signal_plot(pv, S_method2_expand, "Flow") # Change the latter parameter name (method 1 ~ 4)



# Thesis 2 (Sigmoid function)
# Spliting array
num_parts = 7 # Num of parts (Seems like 7 days data)
# Use array_split to split into approximately equal parts
pv_split = np.array_split(pv, num_parts)
co_split = np.array_split(co, num_parts)

days = 0

for pv, co in zip(pv_split, co_split):
    days += 1
    sigmoid = Sigmoid(co=co, pv=pv)
        # sigmoid.delta_pv_op_plot(name= loop_name)

    #Stiction detect
    is_stiction, r_value = sigmoid.detect_stiction()
    
    #Produce output
    print(f"Control valve r value is (The {days} days= {r_value:.2f})")



# Create time data with the same length of pv and co 
# t = np.arange(1, len(pv) + 1)

# Thesis 3 (Linear-Regression)
"""fit linear_regression model"""
# linear_regression = Linear_Rregression(pv=pv, co=co, t=t, draw_log_plot=1)
# linear_regression.detect_stiction()
# linear_regression.store_csv("flow")

"""stic_band quantification"""
# s = np.loadtxt('new_project/csv/essay_3/stiction_signal_linear_regression_flow.csv', delimiter=',')
# s_expand =  np.repeat(s, 6)
# quantificatioin = Quantificatioin(co = co, s = s)
# quantificatioin.perform()















