"""fitting data to svr model"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from new_project.final.function_file import aggregate_points, train_test_split, partial_shuffle
from new_project.thesis_2.sigmoid_class import Sigmoid
import joblib
import scipy.io
import matplotlib.pyplot as plt

np.random.seed(30)


class Svr():
    def __init__(self, file_name : str, data_col : list[list[str]], c_start: float, c_end: float, 
                 epsilon_start: float, epsilon_end: float) -> None:
        self.data_col = np.array(data_col)
        self.df = self._load_data(file_name)
        self.c_s = c_start
        self.c_e = c_end
        self.epsilon_s = epsilon_start
        self.epsilon_e = epsilon_end
    # when instance the class all the file would be loaded
    def _load_data(self, file_path : str):
        # Load data 
        # Load the xlsx file
        df = pd.read_excel(file_path, header = 1, engine="openpyxl") # Header set to 1 for header in the second row and engine 
                                                                    # is to make sure compatibility with .xlsx files.
                                                                    # need to install openpyxl
        return df
    
    # Short for loop used in train_valid
    def _short_for_loop(self, num_windows: int, window_size: int, pv_train: np.ndarray, op_train: np.ndarray, 
                        X_train: list, y_train: list)-> None:
        # Loop through the data to create windows of 60 samples
        for i in range(num_windows):
            # Extract a window of 60 samples from both PV and OP
            pv_window = pv_train[i*window_size:(i+1)*window_size]
            op_window = op_train[i*window_size:(i+1)*window_size]
            # Every x steps take a sample to from a input data
            pv_window = aggregate_points(pv_window)
            op_window = aggregate_points(op_window)
            # Detect stiction and get r_value for this window
            sigmoid = Sigmoid(co=op_window, pv=pv_window)
            _, r_value = sigmoid.detect_stiction()

            # Concatenate PV and OP window into a single input vector of size 40
            input_vector = np.concatenate((sigmoid.pv_scale, sigmoid.op_scale))  # (20, ) + (20, ) = (40, )
            
            # Store the input vector and target r_value
            X_train.append(input_vector)
            y_train.append(r_value)

    # Main algorithm
    def train_valid(self, store: bool, store_path : str)-> None:
        # Create list for store epsilon, c, validatioin mse
        # Create y_train(r_val) and prediction(pred r_val) for all the loop 
        self.epsilon_list, self.valid_mean_squared_error, self.c_list = [], [], []
        self.y_train_all, self.predictions_all_train = [], []
        self.y_valid_all, self.predictions_all_valid = [], []

        # every possible c and epsilon parameter for SVR model
        for c in np.arange(self.c_s, self.c_e, 0.2):
            for epsilon in np.arange(self.epsilon_s, self.epsilon_e, 0.05):
                # Print now the loop is going 
                print(f"c and episolon: {c:.2f} and {epsilon:.2f}")
                # Model construction
                svr_model = SVR(kernel='rbf', C = c, epsilon=epsilon)

                # Initialize arrays to store the input vectors and target values
                X_train, y_train = [], []
                X_valid, y_valid = [], []
                # Loop each control loop
                # pv_col is the column where the PV at and co_col is the column where the CO at 
                for inner_list in self.data_col:
                    pv = self.df[inner_list[0]].to_numpy().flatten()  # Access first element
                    op = self.df[inner_list[1]].to_numpy().flatten()  # Access second element
                    
                    pv_train, op_train, pv_valid, op_valid, _, _ = train_test_split(pv = pv, op = op, valid_size = 0.1, test_size = 0.2)
                    
                    #*Process testing Data
                    # Parameters
                    window_size = 60  # Size of each window
                    num_samples = pv_train.size  
                    num_windows = num_samples // window_size  # Number of windows

                    self._short_for_loop(num_windows, window_size, pv_train, op_train, X_train, y_train)

                    #* Process Validation Data
                    num_samples_valid = pv_valid.size
                    num_windows_valid = num_samples_valid // window_size  # Number of windows

                    self._short_for_loop(num_windows_valid, window_size, pv_valid, op_valid, X_valid, y_valid)

                """training the model"""
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                # Partial suffle train and test data
                X_train, y_train = partial_shuffle(X_train, y_train, 15)
                # Train the SVR model on all the windows
                svr_model.fit(X_train, y_train)
                # Evaluate the model
                predictions = svr_model.predict(X_train)
                mse = mean_squared_error(y_train, predictions)
                print(f"Mean Squared Error: {mse:.5f}")
                # Append into list to see the actual and pred r_val
                self.y_train_all.append(y_train)
                self.predictions_all_train.append(predictions)

                """valid the model"""
                # Use model to valiation data and compute MSE
                predictions_valid = svr_model.predict(X_valid)
                mse_valid = mean_squared_error(y_valid, predictions_valid)
                print(f"Mean Squared Error (Validation): {mse_valid:.5f}")
                # Append into list to compare the actual and pred r_val
                self.y_valid_all.append(y_valid)
                self.predictions_all_valid.append(predictions_valid)
                """append into list to plot the validation mse if needed"""
                self.valid_mean_squared_error.append(round(mse_valid, 5))
                self.epsilon_list.append(epsilon)
                self.c_list.append(c)
                print(f"c : {c}")
                
                # Save the model or not
                if store == 1:
                    joblib.dump(svr_model, store_path)
    
    def draw_validation_mse(self, show: bool)->None:
        """
        Plotting 3d surf with x-axis epsilon hyperparameter, y axis c hyperparameter
        show : bool whether to print the epsilon array and c if or not
        """
        # Store as csv
        epsilon_array = np.array(self.epsilon_list)
        c_array = np.array(self.c_list)
        valid_mean_squared_error = np.array(self.valid_mean_squared_error)
        # Print the epsilon array and c if or not
        if show == 1:
            print(f"epsilon_array : \n{epsilon_array}\n")
            print(f"c_array : \n{c_array}\n")
        
        """MSE plot"""
        # Plot 3d with x = epsilon, y = c, and z = validation MSE 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(epsilon_array, c_array, valid_mean_squared_error)
        # Mark specific points
        highlight_points = [
            (0.1, 0.1),  # (epsilon, C)
            (0.1, 0.5),
            (0.1, 1.1)
        ]
        for epsilon, C in highlight_points:
            # Find the corresponding MSE value for each (epsilon, C) pair
            idx = np.where((epsilon_array == epsilon) & (c_array == C))
            if idx[0].size > 0:  # Check if index exists
                mse_value = valid_mean_squared_error[idx][0]
                ax.scatter(epsilon, C, mse_value, color='red', s=50, label=f"({epsilon}, {C})")
        # Labels for axes
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("C")
        ax.set_zlabel("MSE")
        ax.set_title("3D Plot of MSE varying with C and Epsilon")
        plt.show()
        
    def draw_r_val(self):
        """
        Draw train and valid actual r_val and pred r_val
        """
        # Convert list into array
        self.y_train_all = np.array(self.y_train_all)
        self.y_valid_all = np.array(self.y_valid_all)
        self.predictions_all_train = np.array(self.predictions_all_train)
        self.predictions_all_valid = np.array(self.predictions_all_valid)
        print(f"shape of y_train_all : {self.y_train_all.shape} and y_valid_all : {self.y_valid_all.shape}")
        # Combine y_train_all and y_valid_all
        y_all = np.concatenate([self.y_train_all, self.y_valid_all], axis = 1)
        # Combine predictions_all_train and predictions_all_valid
        prediction_all = np.concatenate([self.predictions_all_train, self.predictions_all_valid], axis = 1)
        # Get the len of y_train
        len_of_y_train = self.y_train_all.shape[1]
        """Acutal and pre R val plot"""
        plt.figure()
        # Actual r_val of all training and validation data
        plt.plot(range(len(y_all.T)), y_all.T, label="Actual r_val")
        # Prediction r_val of all training and validation data
        plt.plot(range(len(prediction_all.T)), prediction_all.T, label="Prediction r_val")
        # Draw a vertical line to seperate training and testing data
        # plt.axvline(x=len_of_y_train, color='r', label='Vertical Line', linewidth=2, linestyle='--')
        # Plotting detail setting
        plt.xlabel("$Windows$")
        plt.ylabel("$R$ $value$")
        plt.title("Training and validation acutal and predicted R value comparison")
        plt.legend()
        # plt.show()
        # Save fig for create gif
        plt.savefig(f"c_{self.c_s}_and_epsilon_{self.epsilon_s}.png")
