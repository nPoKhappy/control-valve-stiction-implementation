import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class Method4():
    def __init__(self, pv : np.ndarray, op : np.ndarray, ws=3, alpha = 0.05) -> None:
        self.pv = pv
        self.op = op
        self.ws = ws
        self.alpha = alpha
        self.s = None


    def perform(self):
        def reverse_arrangement_test(data: np.ndarray) -> int:
            """
            Calculate the reverse arangements A
            :param data: the data sequence need to be detected 
            :return: reverse arangements A
            """
            A = 0
            M = len(data)
            for i in range(M):
                for j in range(i + 1, M):
                    if data[i] > data[j]:
                        A += 1
            return A

        def calculate_AUL_ALL(M: int, alpha: float) -> np.float64:
            """
            Calculate the reverse arangements upper and lower limit
            :param M: window size ws
            :param alpha: significance level
            :return: AUL, ALL
            """
            mu_A = M * (M - 1) / 4
            sigma_A = np.sqrt((2*M**3 + 3*M**2 -5*M )/ 72)
            z_alpha = stats.norm.ppf(1 - alpha/2 )  # Two-tailed test
            # Interval value = z_alpha * sigma_A
            # Lower: mean - interval value
            # Upper: mean + interval value
            # https://stackoverflow.com/questions/60699836/how-to-use-norm-ppf (Explanation look here)
            AUL = mu_A + z_alpha * sigma_A - 0.5 
            ALL = mu_A - z_alpha * sigma_A + 0.5
            return AUL, ALL

        
        
        N = int(np.round(len(self.pv) / self.ws))
        S = np.zeros(N)
        # Same for PV(Process varible) and CO(Controller output)
        AUL, ALL = calculate_AUL_ALL(self.ws, self.alpha)
        # Initilized list to PV and CO A values
        A_PV_values = []
        A_OP_values = []

        for i in range(N):
            PV_window = self.pv[i*self.ws:(i+1)*self.ws]
            OP_window = self.op[i*self.ws:(i+1)*self.ws]
            
            APV = reverse_arrangement_test(PV_window)
            AOP = reverse_arrangement_test(OP_window)
            A_PV_values.append(APV)
            A_OP_values.append(AOP)


            if (ALL < APV < AUL) and not (ALL < AOP < AUL):
                S[i] = 1
            else:
                S[i] = 0
                
        # Convert list to np.array for plotting
        A_PV_values = np.array(A_PV_values)
        A_OP_values = np.array(A_OP_values)


        # Calculate the stiction metric
        Omega = np.sum(S) / N
        # Assign to self.S
        self.s = S

        if Omega > 0:
            print(f"Stiction detected with Omega = {Omega}")
        else:
            print("No stiction detected (Omega = 0)")
        

        # Plotting the results
        plt.figure(figsize=(8, 6))
        plt.plot(A_PV_values, label='A')
        plt.axhline(y=AUL, color='magenta', linestyle='-', label='Upper limit')
        plt.axhline(y=ALL, color='green', linestyle='-', label='Lower limit')
        plt.xlabel('Windows')
        plt.ylabel('A')
        plt.title('Total number of reverse arrangements found in PV each window')
        plt.legend(loc='upper right')
        plt.ylim([-1, max(A_PV_values) + 1])
        plt.show()


        # Plotting the results
        plt.figure(figsize=(8, 6))
        plt.plot(A_OP_values, label='A')
        plt.axhline(y=AUL, color='magenta', linestyle='-', label='Upper limit')
        plt.axhline(y=ALL, color='green', linestyle='-', label='Lower limit')
        plt.xlabel('Windows')
        plt.ylabel('A')
        plt.title('Total number of reverse arrangements found in CO each window')
        plt.legend(loc='upper right')
        plt.ylim([-1, max(A_OP_values) + 1])
        plt.show()

    def store_csv(self, name: str) ->None:
        """store stiction to csv"""
        # Store stiction as csv file
        np.savetxt(f"new_project/csv/essay_1/stiction_signal_method4_{name}.csv", self.s, delimiter=',')