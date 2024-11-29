import numpy as np
import scipy.io




class Quantificatioin():
    def __init__(self, op: np.ndarray, s: np.ndarray, w=6) -> None:
        self.op = op
        self.s = s
        self.w = w
    def perform(self):
        k = 0 # Indicate now is stick or not, a judgement condtion 
        d = [] # Store the current sitck window OP value
        stic_band = [] # abs(d[0] - d[-1]) value would be appened into 

        # Iterate through each window
        for i in range(len(self.s)):
            if i == 0:
                # Handle first window separately
                if self.s[i] > 0:  # Valve is sticky in the first window
                    k = 1
                    d.extend(self.op[0:self.w])  # Add OP values from the first window to 'd'
                else:
                    k = 0
                    d = []  # No stiction, 'd' remains empty
            else:
                if k == 1 and self.s[i] > 0:  # Valve is still sticky
                    k = 1
                    d.extend(self.op[(i) * self.w:(i + 1) * self.w])  # Continue adding OP values
                elif k == 1 and self.s[i] == 0:  # Valve just overcame stiction
                    k = 0
                    if d:  # Ensure 'd' is not empty before calculating stiction band
                        stic_band.append(abs(d[0] - d[-1]))
                    else:
                        raise ValueError(f"the iteration {i} d is empty.")
                    d = []  # Reset 'd' for the next potential sticky phase
                elif k == 0 and self.s[i] == 0:  # Valve is freely moving
                    k = 0
                    d = []  # Keep 'd' empty
                elif k == 0 and self.s[i] > 0:  # Valve sticks again
                    k = 1
                    d.extend(self.op[(i) * self.w:(i + 1) * self.w])  # Start collecting OP values again

        if d:  # If there's leftover data in 'd', compute the final stiction band
            stic_band.append(abs(d[0] - d[-1]))
            
        # Approximation for stiction band
        Psi = max(stic_band) if stic_band else 0

        print("Approximation for stiction band (Ψ):", Psi)
        


