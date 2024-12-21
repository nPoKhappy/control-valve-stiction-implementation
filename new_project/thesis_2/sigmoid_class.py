import numpy as np
from scipy.optimize import  least_squares
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import Union

# Modified piecewise linear function
def step_function(x):
    return np.piecewise(x, 
                        [x < 0, x >= 0], 
                        [lambda x: np.clip(0.15 * x + 0.3, 0, 1),  # left side restricted to [0, 1]
                         lambda x: np.clip(0.15 * x + 0.7, 0, 1)])  # right side

class Sigmoid():
    def __init__(self, op: np.ndarray, pv: np.ndarray) -> None:
        self.op = op
        self.pv = pv

    # Define the sigmoid function 
    def _sigmoid(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        # Prevent overflow(the parameter a or b pass into this function being too large)
        z = a * x + b
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))
    def _modified_step_function(self, x: np.ndarray, s_a: float, s_b: float, a: float, b: float) -> np.ndarray:
        a = min(a, 0.5)
        b = min(b, 0.5)
        return np.piecewise(x, 
                            [x < 0, x >= 0], 
                            [lambda x: np.clip(s_a * x + a, 0, 1),  # left side restricted to [0, 1]
                            lambda x: np.clip(s_b * x + b, 0, 1)])  # right side restricted to [0, 1]
    
    # Calculate correlation coefficient
    def _calculate_correlation_coefficient(self, y, y_fit):
        """
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient <- wikipedia explain
        Pearson correlation coefficient
        pho_x_y = cov(x, y) / sigma_x * sigma_y
        """
        if np.all(y_fit == y_fit[0]):
            print("Warning: 'y_fit' is constant; setting r_value to 0.")
            return 0
        
        return pearsonr(y, y_fit)[0] # The first element from result is the statistic and the sec one is p_value

    # Main function for detecting stiction
    def detect_stiction(self, r_threshold=0.5) -> Union[bool, float]: 
        # Second step：fit Sigmoid function to ΔPV and OP
        """
        xdata is the independent variable want to fit in
        and ydata is the dependent variable to fit in
        the model is the sigmoid function just defined previsosly
        popt is the a and b parameters
        _ is the unused covariance matrix
        """
        # Residuals function for least_squares (difference between predicted and observed data)
        def residuals(params, x, y):
            a, b = params
            return self._sigmoid(x, a, b) - y
        # Initial guess for the parameters [a, b]
        initial_guess = [1, 0]
        # Use least_squares to fit the sigmoid curve
        result = least_squares(residuals, initial_guess, args=(self.pv, self.op), method='trf')
        # Extract fitted parameters
        a_fitted, b_fitted = result.x
        # print(f"para s_a : {s_a_fitted}, s_b : {s_b_fitted}, a : {a_fitted} and b : {b_fitted}\n")
        # Compute the fitted y values
        y_fit = self._sigmoid(self.pv, a_fitted, b_fitted)
        # Third step: calculate the correlation coefficient
        r_value = self._calculate_correlation_coefficient(self.op, y_fit)
        # Round to dicimal 2
        r_value = round(r_value, 2)
        # If pearson coeff less than zero, it means negative relationship
        if r_value < 0:
            r_value = 0.00
        # Is there a stiction happened
        is_stiction = r_value >= r_threshold

        return is_stiction, r_value


    def delta_pv_op_plot(self, name: str, filename: str) -> None:
        """delta pv v.s op plot"""
        # flatten array from 2d -> 1d
        op = self.co.flatten()
        pv = self.pv.flatten() 


        op_norm = (op - np.min(op)) / (np.max(op) - np.min(op))
        delta_pv = np.diff(pv, prepend=pv[0])

        mean_pv = np.mean(delta_pv)
        std_pv = np.std(delta_pv)
        pv_normalized = (delta_pv - mean_pv) / std_pv
        
        
        # Create a range of x values
        x = np.linspace(-4, 4, 400)
        # Plot the step function with slopes adjusted
        y = step_function(x)


        plt.figure(dpi=150)
        plt.plot(x, y, label='Adjusted Step Function', color='black')
        plt.scatter(pv_normalized, op_norm, s=7)
        plt.title(f"$Chemical$ $loop{name}$")
        plt.xlabel(r"$\Delta PV(k)$")
        plt.ylabel(r"$CO(k)$")
        # plt.show()

        # Save the plot as an image file (PNG)
        plt.savefig(filename)
        # plt.close()  # Close the plot to avoid memory issues

