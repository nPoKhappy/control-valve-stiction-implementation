
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt



class Method1():
    def __init__(self, pv : np.ndarray, op : np.ndarray, ws=8, alpha = 0.05) -> None: #window size and significance level
        self.pv = pv
        self.op = op
        self.ws = ws
        self.alpha = alpha
        self.s = None

    def perform(self) -> None:
        # Number of data windows
        N = int(np.round(len(self.pv) / self.ws))

        # Reshape the data into windows (8 sizes for a small sample)
        PV_windows = np.array_split(self.pv, N)
        OP_windows = np.array_split(self.op, N)

        # Calculate variance by manual
        def manual_variance(data):
            # Calculate the mean
            mean = sum(data) / len(data)
            
            # Calculate the sum of squared differences from the mean
            squared_diffs = [(x - mean) ** 2 for x in data]
            
            # Calculate variance (using Bessel's correction with n-1)
            variance = sum(squared_diffs) / (len(data) - 1)
            
            return variance


        # F-test function
        def f_test(segment1, segment2):
            var1 = manual_variance(segment1)
            var2 = manual_variance(segment2)
            F = var1 / var2
            f_critical = f.ppf(1 - self.alpha, len(segment1) - 1, len(segment2) - 1)
            return F, f_critical

        # Calculate the process varible i and i+1 window,if steady-state => True 
        # Calculate the cotroller output i and i+1 window,if transient => True 
        # If first(F_value < F_critical) and second(F_value > F_critical) 
        # Sticiton detected
        S = np.zeros(N)

        f_pv_list = []
        f_op_list = []


        for i in range(N - 1):
            F_pv, critical_pv = f_test(PV_windows[i], PV_windows[i + 1])
            F_op, critical_op = f_test(OP_windows[i], OP_windows[i + 1])
            f_pv_list.append(F_pv)
            f_op_list.append(F_op)
            f_pv_critical = critical_pv
            f_op_critical = critical_op
            if (F_pv < critical_pv and F_op > critical_op):
                S[i], S[i+1] = 1, 1
            else:
                S[i], S[i+1] = 0, 0

        # Assign to self.s
        self.s = S
        # Calculate omega
        Omega = np.sum(S) / N

        if Omega > 0:
            print(f"Stiction detected with Omega = {Omega}")
        else:
            print("No stiction detected (Omega = 0)")



        # Number of windows
        windows = range(len(f_pv_list))
        """ PV """
        # Plotting
        plt.figure()

        # Plot F-values
        plt.plot(windows, f_pv_list, label="F-value")

        # Plot critical F-value (either a constant or an array)
        if isinstance(f_pv_critical, (int, float)):
            plt.axhline(y=f_pv_critical, color='r', linestyle='-', label="Critical F-value")
        else:
            plt.plot(windows, f_pv_critical, color='r', linestyle='-', label="Critical F-value")

        # Labels and title
        plt.xlabel("Windows")
        plt.ylabel("F-value")
        plt.title("F-value for PV ")
        plt.legend()
        plt.show()

        """OP"""
        # Plotting
        plt.figure()

        # Plot F-values
        plt.plot(windows, f_op_list, label="F-value")

        # Plot critical F-value (either a constant or an array)
        if isinstance(f_op_critical, (int, float)):
            plt.axhline(y=f_op_critical, color='r', linestyle='-', label="Critical F-value")
        else:
            plt.plot(windows, f_op_critical, color='r', linestyle='-', label="Critical F-value")

        # Labels and title
        plt.xlabel("Windows")
        plt.ylabel("F-value")
        plt.title("F-value for CO ")
        plt.legend()
        plt.show()

    def store_csv(self, name: str):
        """store stiction to csv"""
        # Store stiction as csv file
        np.savetxt(f"new_project/csv/essay_1/stiction_signal_method1_{name}.csv", self.s, delimiter=',')