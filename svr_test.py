"""test trained svr model"""

import numpy as np
from sklearn.metrics import mean_squared_error
from new_project.final.function_file import aggregate_points, train_test_split, partial_shuffle
from new_project.thesis_2.sigmoid_class import Sigmoid
from new_project.thesis_1.plot import plot_pv_sp_op
import joblib
import scipy.io
import time


np.random.seed(30)



# Load the model using pickle
path_file = "new_project/csv/final/svr_model/svr_model_best.pkl"
svr_model = joblib.load(path_file)

# Load the .mat file
mat_contents = scipy.io.loadmat('new_project/isdb10.mat')

# Access the 'cdata' structure within 'mat_contents'
cdata = mat_contents['cdata'][0,0]

# Access the 'chemicals' structure within 'cdata'
chemicals = cdata['chemicals'][0, 0]
pulpPapers = cdata['pulpPapers'][0, 0]

loop_list = [1, 2, 3, 6, 10, 11, 12, 13, 23, 24, 29, 32]
loop_array = np.array(loop_list)


# Load the overall stiction R in every loop
stiction_array = np.loadtxt('new_project/csv/final/stiction_array.csv', delimiter=',')

# Index for stiction array
stiction_index = 0

# Initialize arrays to store the input vectors and target values
X_train = []
y_train = []
diff_r_val_list = []

for loop_name in loop_array:
    # Take overall r val 
    overall_r_val = stiction_array[stiction_index]
    stiction_index += 1
    # Access a specific loop
    loop = chemicals[f'loop{loop_name}'][0, 0]
    # Now you can access fields within 'loop2'
    pv = loop['PV'].flatten() # shape (samples num, )
    op = loop['OP'].flatten() # shape (samples num, )
    _, _, pv_valid, op_valid, pv_test, op_test = train_test_split(pv, op, 0.1, 0.2)
    # Parameters
    window_size = 60  # Size of each window
    num_samples = pv_valid.size  
    num_windows = num_samples // window_size  # Number of windows
    # Loop through the data to create windows of 60 samples
    for i in range(num_windows):
        # Pause for 1 sec
        time.sleep(1)
        # Extract a window of 60 samples from both PV and OP
        pv_window = pv_valid[i*window_size:(i+1)*window_size]
        op_window = op_valid[i*window_size:(i+1)*window_size]
        # Every x step take sample to form new input data
        pv_window = aggregate_points(pv_window)
        op_window = aggregate_points(op_window)
        # Detect stiction and get r_value for this window
        sigmoid = Sigmoid(op=op_window, pv=pv_window)
        _, r_value = sigmoid.detect_stiction()
        # Plot PV and OP 
        # plot_pv_sp_op(pv = sigmoid.pv_scale, sp = None, op = sigmoid.op_scale, loop_name = loop_name)
        # Compare with overall r_val 
        diff_r_val = overall_r_val - r_value
        if abs(diff_r_val) >= 0.15:
            diff_r_val_list.append(abs(round(diff_r_val, 2)))
            continue # Continue to next window
        

        # Concatenate PV and OP window into a single input vector of size 40
        input_vector = np.concatenate((sigmoid.pv_scale, sigmoid.op_scale))  # (20, ) + (20, ) = (40, )
        
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
print(f"y_train : {y_train}\n, predictions :\n {predictions}\n")
print(f"Mean Squared Error: {mse}")
y_combine = np.column_stack((y_train.T,predictions.T))
# np.savetxt(f"new_project/csv/final/svr_model/r_val_svr_test.csv", y_combine, header="Real r_val, Pre r_val", delimiter=',', fmt="%.3f")




