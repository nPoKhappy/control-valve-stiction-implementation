import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt



class Method2():
    def __init__(self, pv : np.ndarray, op : np.ndarray, ws=8, alpha = 0.05) -> None:
        self.pv = pv
        self.op = op
        self.ws = ws
        self.alpha = alpha  
        self.s = None
        

    def perform(self)-> None:
        
        # Number of data windows
        N = int(np.round(len(self.pv) / self.ws))

        # Reshape the data into windows
        PV_windows = np.array_split(self.pv, N)
        OP_windows = np.array_split(self.op, N)

        # t-test function
        def t_test(segment1, segment2):
            mean1 = np.mean(segment1)
            mean2 = np.mean(segment2)
            std1 = np.std(segment1, ddof=1)
            std2 = np.std(segment2, ddof=1)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(segment1) - 1) * std1**2 + (len(segment2) - 1) * std2**2) / (len(segment1) + len(segment2) - 2))
            
            # Calculate the t-value
            t_value = np.abs(mean1 - mean2) / (pooled_std * np.sqrt(1/len(segment1) + 1/len(segment2)))
            
            # Degrees of freedom
            df = len(segment1) + len(segment2) - 2
            
            # Critical t-value
            t_critical = t.ppf(1 - self.alpha , df)  # Two-tailed test
            
            return t_value, t_critical

        # Initialize the stiction 
        S = np.zeros(N)
        t_pv_list = []
        t_op_list = []

        for i in range(N - 1):
            t_pv, t_pv_critical = t_test(PV_windows[i], PV_windows[i + 1])
            t_op, t_op_critical = t_test(OP_windows[i], OP_windows[i + 1])
            t_pv_list.append(t_pv)
            t_op_list.append(t_op)
            
            if (t_pv < t_pv_critical and t_op > t_op_critical):
                S[i], S[i + 1] = 1, 1
            else:
                S[i], S[i + 1] = 0, 0

        # Calculate the stiction metric
        Omega = np.sum(S) / N
        # Assign to self.S
        self.s = S

        if Omega > 0:
            print(f"Stiction detected with Omega = {Omega}")
        else:
            print("No stiction detected (Omega = 0)")


        # Number of windows
        windows = range(len(t_pv_list))

        """PV"""
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot F-values
        plt.plot(windows, t_pv_list, label="tp-value")

        # Plot critical F-value (either a constant or an array)
        if isinstance(t_pv_critical, (int, float)):
            plt.axhline(y=t_pv_critical, color='r', linestyle='-', label="Critical tp-value")
        else:
            plt.plot(windows, t_pv_critical, color='r', linestyle='-', label="Critical tp-value")

        # Labels and title
        plt.xlabel("Windows")
        plt.ylabel("t-value")
        plt.title("t-value for PV ")
        plt.legend()
        plt.close()

        """OP"""
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot F-values
        plt.plot(windows, t_op_list, label="tp-value")

        # Plot critical F-value (either a constant or an array)
        if isinstance(t_op_critical, (int, float)):
            plt.axhline(y=t_op_critical, color='r', linestyle='-', label="Critical tp-value")
        else:
            plt.plot(windows, t_op_critical, color='r', linestyle='-', label="Critical tp-value")

        # Labels and title
        plt.xlabel("Windows")
        plt.ylabel("t-value")
        plt.title("t-value for CO ")
        plt.legend()
        plt.close()
        
    def store_csv(self, name: str):
        """store stiction to csv"""
        # Store stiction as csv file
        np.savetxt(f"new_project/csv/essay_1/stiction_signal_method2_{name}.csv", self.s, delimiter=',')