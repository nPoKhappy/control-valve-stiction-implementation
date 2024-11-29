"""fitting data to svr model"""

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from new_project.final.function_file import aggregate_points, train_test_split, partial_shuffle
from new_project.thesis_2.sigmoid_class import Sigmoid
import joblib
import scipy.io
import matplotlib.pyplot as plt

np.random.seed(30)


class Svr():
    def __init__(self, loop_list: list, c_start: float, c_end: float, 
                 epsilon_start: float, epsilon_end: float) -> None:
        self.loop_array = np.array(loop_list)
        self.chemicals, self.stiction_array = self._load_data()
        self.c_s = c_start
        self.c_e = c_end
        self.epsilon_s = epsilon_start
        self.epsilon_e = epsilon_end
    # when instance the class all the file would be loaded
    def _load_data(self):
        # Load the .mat file
        mat_contents = scipy.io.loadmat('new_project/isdb10.mat')
        # Access the 'cdata' structure within 'mat_contents'
        cdata = mat_contents['cdata'][0,0]
        # Access the 'chemicals' structure within 'cdata'
        chemicals = cdata['chemicals'][0, 0]
        # Load the overall stiction R in every loop
        stiction_array = np.loadtxt('new_project/csv/final/stiction_array.csv', delimiter=',')
        
        return chemicals, stiction_array
    
    # Short for loop used in train_valid
    def _short_for_loop(self, num_windows: int, window_size: int, pv_train: np.ndarray, op_train: np.ndarray, 
                        overall_r_val: np.float64, X_train: list, y_train: list)-> None:
        # Loop through the data to create windows of 60 samples
        for i in range(num_windows):
            # Extract a window of 60 samples from both PV and OP
            pv_window = pv_train[i*window_size:(i+1)*window_size]
            op_window = op_train[i*window_size:(i+1)*window_size]
            # Every x steps take a sample to from a input data
            pv_window = aggregate_points(pv_window)
            op_window = aggregate_points(op_window)
            # Detect stiction and get r_value for this window
            sigmoid = Sigmoid(op=op_window, pv=pv_window)
            _, r_value = sigmoid.detect_stiction()
            
            # Compare windowed r_value with overall r_val (Kick outliers)
            diff_r_val = overall_r_val - r_value
            if abs(diff_r_val) >= 0.5:
                continue # Continue to next window if difference is too large

            # Concatenate PV and OP window into a single input vector of size 40
            input_vector = np.concatenate((sigmoid.pv_scale, sigmoid.op_scale))  # (20, ) + (20, ) = (40, )
            
            # Store the input vector and target r_value
            X_train.append(input_vector)
            y_train.append(r_value)

    # Main algorithm
    def train_valid(self, store: bool)-> None:
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

                # Index for stiction array
                stiction_index = 0

                # Initialize arrays to store the input vectors and target values
                X_train, y_train = [], []
                X_valid, y_valid = [], []
                # Loop each chemical loop
                for loop_name in self.loop_array:
                    # Get overall r val 
                    overall_r_val = self.stiction_array[stiction_index]
                    stiction_index += 1
                    # Access a specific loop
                    loop = self.chemicals[f'loop{loop_name}'][0, 0]
                    # Now you can access fields within 'loop2'
                    pv = loop['PV'].flatten() # shape (samples num, )
                    op = loop['OP'].flatten() # shape (samples num, )
                    pv_train, op_train, pv_valid, op_valid, _, _ = train_test_split(pv, op, 0.1, 0.2)
                    
                    #*Process testing Data
                    # Parameters
                    window_size = 60  # Size of each window
                    num_samples = pv_train.size  
                    num_windows = num_samples // window_size  # Number of windows

                    self._short_for_loop(num_windows, window_size, pv_train, op_train, overall_r_val, X_train, y_train)

                    #* Process Validation Data
                    num_samples_valid = pv_valid.size
                    num_windows_valid = num_samples_valid // window_size  # Number of windows

                    self._short_for_loop(num_windows_valid, window_size, pv_valid, op_valid, overall_r_val, X_valid, y_valid)

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
                    joblib.dump(svr_model, 'new_project/csv/final/svr_model/svr_model_best.pkl')
    
    def draw_validation_mse(self, show: bool)->None:
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
        
    # Draw train and valid actual r_val and pred r_val
    def draw_r_val(self):
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
