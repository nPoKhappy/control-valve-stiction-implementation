import numpy as np
import matplotlib.pyplot as plt


class Linear_Rregression():
    def __init__(self, pv: np.ndarray, op: np.ndarray, t: np.ndarray,draw_log_plot: bool, w=6, n=15) -> None:
        self.pv = pv.flatten()
        self.op = op.flatten()
        self.t = t.flatten()
        self.w = w
        self.n = n
        self.draw = draw_log_plot

    def _fitlinear(self, tdata: np.ndarray, ydata: np.ndarray) -> np.float64:
        """
        Fit a linear model y = mt + b
        explain here https://blog.csdn.net/weixin_43544164/article/details/122350501
        """
        A = np.vstack([tdata, np.ones(len(tdata))]).T
        m, b = np.linalg.lstsq(A, ydata, rcond=None)[0] # Use least square to get the linear regression model
        return m, b

    def detect_stiction(self) -> None:
        N = len(self.pv)
        S = []
        a = 0
        slope_ratio_list = []
        # Calculate the slope_ratio with linear_regression and compared to threshold
        for i in range(N // self.w):
            PVdata = self.pv[a:a+self.w]  # Splitting PV signal into non-overlapping and equal-width windows
            OPdata = self.op[a:a+self.w]  # Splitting OP signal into non-overlapping and equal-width windows
            tdata = self.t[a:a+self.w]

            mp, bp = self._fitlinear(tdata, PVdata)  # Constructing linear model for the ith window of PV signal
            mo, bo = self._fitlinear(tdata, OPdata)  # Constructing linear model for the ith window of OP signal
            slope_ratio = abs(mo / mp)
            slope_ratio_list.append(slope_ratio)
            
            # Added abs to make sure only care about the ration not influenced by negative value
            if slope_ratio >= self.n:  # Verifying if the slope ratio exceeds the threshold
                S.append(1)  # Stiction signal becomes one if the valve sticks
            else:
                S.append(0)

            a = (i + 1) * self.w
        # Assign to self.s just in order to use in store_csv method
        self.s = np.array(S)
        
        # Computing stiction index
        stiction_index = sum(S) / (N // self.w) * 100
        
        if stiction_index > 0:
            print(f"Given control valve is sticky. Stiction index is {stiction_index:.3f}.")
        else:
            print(f"Given control valve is not sticky.")
        
        
        """draw logn(threshold) v.s. logR(slope_ratio)"""
        if self.draw == 1:
            # Convert to array_like and make it to log form
            slope_ratio_array = np.array(slope_ratio_list)
            log_R = np.log(slope_ratio_array)
            eta = np.array([self.n])
            log_eta = np.log(eta)

            # Create the plot
            plt.figure(figsize=(8, 6))
            plt.plot(range(len(log_R)), log_R, label='log(R)')
            plt.axhline(y=log_eta, color='r', linestyle='--', label='log(η)')

            # Add labels and legend
            plt.xlabel('Window')
            plt.legend()
            plt.title("log(eta) v.s. log(R) in chem6")
            # Display the plot
            plt.show()


    def store_csv(self, name: str) ->None:
        """store stiction to csv"""
        # Store stiction as csv file
        np.savetxt(f"new_project/csv/essay_3/stiction_signal_linear_regression_{name}.csv", self.s, delimiter=',')




