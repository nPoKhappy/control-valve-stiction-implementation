"""test trained svr model"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error # Used to evaluate model
from new_project.final.function_file import aggregate_points, train_test_split
from new_project.thesis_2.sigmoid_class import Sigmoid
import joblib # Used to load model 
import matplotlib.pyplot as plt



# Load the model using pickle
model_path : str = "new_project/csv/final/svr_model/CVOP_20220201~0430.pkl"
svr_model = joblib.load(model_path)

file_path : str = "new_project/CVOP_20220201~0430.xlsx"
df : pd.DataFrame = pd.read_excel(file_path, header = 1, engine="openpyxl")


data_cols : list[list[str]] = [["FT_107", "FV_107.OUT"], ["FT_117", "FV_117.OUT"], ["FT_174", "FV_174.OUT"], ["FT_181", "FV_181.OUT"],
             ["FT_2004", "FV_2004.OUT"], ["FT_306", "FV_306.OUT"], ["FT_308", "FV_308.OUT"]] # 

# Initialize arrays to store the input vectors and target values
X_train = []
y_train = []

for inner_list in data_cols:
    pv : np.ndarray = df[inner_list[0]].to_numpy().flatten()  # Access first element
    op : np.ndarray = df[inner_list[1]].to_numpy().flatten()  # Access second element
    _, _, _, _, pv_test, op_test = train_test_split(pv, op, 0.1, 0.2)
    # Parameters
    window_size : int = 60  # Size of each window
    num_samples : int = pv_test.size  
    num_windows : int = num_samples // window_size  # Number of windows
    # Loop through the data to create windows of 60 samples
    for i in range(num_windows):
        # First step：calculate ΔPV（first order difference backward of PV）
        delta_pv : np.ndarray = np.diff(pv, prepend = pv[0]) # Prepend to make sure the shape of diff_pv is the same as op
                                            # Add pv[0] to let delta_pv first element become 0
        mean_pv : float= np.mean(delta_pv)
        std_pv : float = np.std(delta_pv)
        pv_normalized : np.ndarray = (delta_pv - mean_pv) / std_pv
        # Make op to [0, 1]
        op_range : float = np.max(op) - np.min(op)
        op_norm : np.ndarray = (op - np.min(op)) / op_range
        # Extract a window of 60 samples from both PV and OP
        pv_window : np.ndarray = pv_normalized[i*window_size:(i+1)*window_size]
        op_window : np.ndarray = op_norm[i*window_size:(i+1)*window_size]
        # If the controller output didnt change wihtin the period window
        # the pearson correlation coeffcient wont work since the denomitor 
        # is standard deviation and its value will be zero 
        if (np.max(op_window) - np.min(op_window) == 0): 
                continue
        # Detect stiction and get r_value for this window
        sigmoid : Sigmoid = Sigmoid(op=op_window, pv=pv_window)
        _, r_value = sigmoid.detect_stiction()
        # Plot PV and OP 
        # plot_pv_sp_op(pv = sigmoid.pv_scale, sp = None, op = sigmoid.op_scale, loop_name = loop_name)
        # Concatenate PV and OP window into a single input vector of size 40
        input_vector : np.ndarray = np.concatenate((sigmoid.pv, sigmoid.op))  # (20, ) + (20, ) = (40, )
        
        # Store the input vector and target r_value
        X_train.append(input_vector)
        y_train.append(r_value)


# Convert lists to numpy arrays for training
X_train = np.array(X_train)
y_train = np.array(y_train)

# Predict using trained model
predictions = svr_model.predict(X_train)

# Evaluating performance
mse = mean_squared_error(y_train, predictions)
print(f"y_train\n : {y_train}\n, predictions :\n {predictions}\n")
print(f"Mean Squared Error: {mse}")
# Draw prediction and actual r value
plt.figure()
plt.plot(range(len(predictions)), predictions, label="predictions")
plt.plot(range(len(y_train)), y_train, label="actual")

plt.legend()
plt.title("test model")
plt.xlabel("$windows$")
plt.ylabel("$R value$")
plt.show()