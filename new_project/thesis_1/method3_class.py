import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt





class Method3():
    def __init__(self, pv : np.ndarray, op : np.ndarray, ws=8, alpha = 0.05) -> None:
        self.pv = pv
        self.op = op
        self.ws = ws
        self.alpha = alpha
        self.s = None

    def perform(self) -> None:
        """Use method 3 draw the PV and CO plot"""
        # Number of data windows
        N = int(np.round(len(self.pv) / self.ws))

        # Reshape the data into windows (8 sizes for a small sample)
        PV_windows = np.array_split(self.pv, N)
        OP_windows = np.array_split(self.op, N)


        def calculate_covariance_matrix(X: np.ndarray, mu: np.float64) -> np.float64:
            """Calculate the covariance matrix A for the data X."""
            # reshape(-1, 1) => one col mutiple rows; reshape(1, -1) => one row mutiple cols 
            A = np.sum([(x - mu).reshape(-1, 1) @ (x - mu).reshape(1, -1) for x in X], axis=0)
            return A

        def modified_hotelling_T2_test(Xi: np.ndarray, Xj: np.ndarray, ws=8, alpha=0.05) -> np.float64:
            """Perform the modified Hotelling T^2 test for two data windows Xi and Xj."""
            # Mean 
            mu_i = np.mean(Xi, axis=0)
            mu_j = np.mean(Xj, axis=0)
            
            # Covariance matrices
            A_i = calculate_covariance_matrix(Xi, mu_i)
            A_j = calculate_covariance_matrix(Xj, mu_j)
            
            # Sample size
            n = len(Xi) - 1  # Assuming Xi and Xj have the same length
            
            # Pooled covariance matrix
            S = (A_i + A_j) / (2 * n)
            # Inverse S
            inverse_S = np.linalg.inv(S)
            # Difference in means
            d = mu_i - mu_j
            
            # Test statistic
            T = 0.5 * ws * d.T @ inverse_S @ d
            
            # Degrees of freedom for the F-distribution
            p = 1  # Since PV and OP are analyzed separately
            
            S_i_critical = A_i / n
            S_j_critical = A_j / n
            S_critical = (S_i_critical + S_j_critical) / 2
            inverse_S_critical = np.linalg.inv(S_critical)
            
            first_term = (d.T @ inverse_S_critical @ S_i_critical @ inverse_S_critical @ d) / (2 * d.T @ inverse_S_critical @ d)
            second_term = (d.T @ inverse_S_critical @ S_j_critical @ inverse_S_critical @ d) / (2 * d.T @ inverse_S_critical @ d)
            A = first_term**2 + second_term**2
            f_B = n / A
            
            # Critical value from the F-distribution
            F_critical = f.ppf(1 - alpha, p, f_B - p + 1)
            Tpf_B_alpha2 = F_critical
            
            return T, Tpf_B_alpha2


        # Lists to store the test values and critical values
        T_pv_values = []
        critical_pv_values = []

        T_op_values = []
        critical_op_values = []

        S = np.zeros(N)

        # Perform the modified Hotelling T^2 test for each pair of windows
        for i in range(N - 1):
            T_pv, critical_pv = modified_hotelling_T2_test(PV_windows[i], PV_windows[i + 1])
            T_op, critical_op = modified_hotelling_T2_test(OP_windows[i], OP_windows[i + 1])
            T_pv_values.append(T_pv)
            T_op_values.append(T_op)
            critical_pv_values.append(critical_pv)
            critical_op_values.append(critical_op)
            if (T_pv < critical_pv and T_op > critical_op):
                S[i], S[i+1] = 1, 1
            else:
                S[i], S[i+1] = 0, 0


        # Calculate the stiction metric
        Omega = np.sum(S) / N
        # Assign to self.S
        self.s = S

        if Omega > 0:
            print(f"Stiction detected with Omega = {Omega}")
        else:
            print("No stiction detected (Omega = 0)")

        # Convert lists to numpy arrays for plotting
        T_pv_values = np.array(T_pv_values)
        critical_pv_values = np.array(critical_pv_values)
        T_op_values = np.array(T_op_values)
        critical_op_values = np.array(critical_op_values)


        # Plotting
        """PV"""
        plt.figure(figsize=(8, 6))
        plt.plot(T_pv_values, label=r'$T^2_{pf_B}$ (test value)')
        plt.plot(critical_pv_values, label=r'$T^2_{pf_B, \alpha}$ (critical value)', color='r')
        plt.xlabel('Windows')
        plt.ylabel(r'$T^2_{pf_B}$')
        plt.legend()
        plt.title(r'$T^2_{pf_B}$ for PV ')
        plt.show()

        """OP"""
        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(T_op_values, label=r'$T^2_{pf_B}$ (test value)')
        plt.plot(critical_op_values, label=r'$T^2_{pf_B, \alpha}$ (critical value)', color='r')
        plt.xlabel('Windows')
        plt.ylabel(r'$T^2_{pf_B}$')
        plt.legend()
        plt.title(r'$T^2_{pf_B}$ for CO ')
        plt.show()
    
    def store_csv(self, name: str) ->None:
        """store stiction to csv"""
        # Store stiction as csv file
        np.savetxt(f"new_project/csv/essay_1/stiction_signal_method3_{name}.csv", self.s, delimiter=',')