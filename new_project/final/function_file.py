"""wrtie some function to use in other files"""
import numpy as np



def split_data(name: str, samples_per_set: int) -> list[np.float64]:
    """
    split data into samples per set
    
    """
    # Load final data
    # Shape is (samples,)
    combined_pv = np.loadtxt(f'new_project/csv/final/{name}_pv.csv', delimiter=',')
    combined_op = np.loadtxt(f'new_project/csv/final/{name}_op.csv', delimiter=',')

    # Calculate the number of complete chunks
    n_complete_chunks = len(combined_pv) // samples_per_set
    max_samples = n_complete_chunks * samples_per_set
    # Only to max_samples range, excluding the leftover 
    combined_pv = combined_pv[:max_samples]
    combined_op = combined_op[:max_samples]

    # Split combined_pv and combined_op into sub-datasets 
    # pv and op split looks like [(), (), ()......()] list contain mutiple array
    pv_split = np.array_split(combined_pv, n_complete_chunks)
    op_split = np.array_split(combined_op, n_complete_chunks)
    return pv_split, op_split


# testing
if __name__ == "__main__":
    pv_split, op_split = split_data(name="13_11_1_10_24", samples_per_set=100)
    for pv in pv_split:
        print(f"{pv.shape}\n")




def train_test_split(pv: np.ndarray, op: np.ndarray,valid_size: float ,test_size: float)-> np.ndarray:
    """
    Parameters:
    pv: process variable
    op: controller output
    valid size: the size of validation data
    test size: the size of test data
    Return:
    pv_train, op_train, pv_valid, op_valid, pv_test, op_test
    """
    # Calculate the train split index
    size_except_train = valid_size + test_size
    train_split_idx = int((1 - size_except_train) * len(pv))
    # Calculate validation split index
    valid_split_idx = int((1 - test_size) * len(pv))
    # Extract train data
    pv_train = pv[:train_split_idx] 
    op_train = op[:train_split_idx]
    # Extract valid data
    pv_valid = pv[train_split_idx:valid_split_idx] 
    op_valid = op[train_split_idx:valid_split_idx]
    # Extract test data
    pv_test = pv[valid_split_idx:]
    op_test = op[valid_split_idx:]
    # Because the data is not enough for train and valid at the same time
    # Use some of the train data to validation
    size_except_train_valid = size_except_train + 0.1
    train_to_valid_split_index = int((1 - size_except_train_valid) * len(pv))
    pv_train_to_valid = pv_train[train_to_valid_split_index:valid_split_idx]
    op_train_to_valid = op_train[train_to_valid_split_index:valid_split_idx]
    # Combine extracted data into valid data
    pv_valid = np.concatenate([pv_train_to_valid, pv_valid])
    op_valid = np.concatenate([op_train_to_valid, op_valid])

    return pv_train, op_train, pv_valid, op_valid, pv_test, op_test



def sample_data(split_pv: np.ndarray,split_op ,num_samples: int) -> np.ndarray:

    # Calculate the maximum starting index to ensure we can get 'num_samples' consecutive elements
    max_start_idx = len(split_pv) - num_samples
    
    # Randomly choose a starting index
    start_idx = np.random.randint(0, max_start_idx + 1)
    # Select consecutive samples
    pv_samples = split_pv[start_idx : start_idx + num_samples]
    op_samples = split_op[start_idx : start_idx + num_samples]
    
    return pv_samples, op_samples



def aggregate_points(input_array: np.ndarray, group_size=3):
    
    sample_array = input_array[::group_size]

    return sample_array

if __name__ == "__main__":
    # Example usage
    input_array = np.arange(60)  # Creating an example array of size 60
    output_array = aggregate_points(input_array)
    print(output_array)


# Do the partial shuffle
def partial_shuffle(arr1: np.ndarray, arr2: np.ndarray, window_size: int)->np.ndarray:
    arr1_shuffled = arr1.copy()
    arr2_shuffled = arr2.copy()

    for i in range(0, len(arr1), window_size):
        end = min(i + window_size, len(arr1))  # 確保不會超過陣列長度
        indices = np.arange(i, end)
        shuffled_indices = np.random.permutation(indices)

        arr1_shuffled[indices] = arr1[shuffled_indices]
        arr2_shuffled[indices] = arr2[shuffled_indices]
    
    return arr1_shuffled, arr2_shuffled


def count_samples_in_ranges(arr: np.ndarray[np.float64])-> np.ndarray[int]:
    # Count samples in (-4, -3) and (3, 4)
    count_neg4_neg3_pos3_pos4 = np.sum((arr > -4) & (arr <= -3)) + np.sum((arr > 3) & (arr <= 4))
    
    # Count samples in (-3, -2) and (2, 3)
    count_neg3_neg2_pos2_pos3 = np.sum((arr > -3) & (arr <= -2)) + np.sum((arr > 2) & (arr <= 3))
    
    # Count samples in (-2, -1) and (1, 2)
    count_neg2_neg1_pos1_pos2 = np.sum((arr > -2) & (arr <= -1)) + np.sum((arr > 1) & (arr <= 2))
    
    # Count samples in (-1, 1)
    count_neg1_pos1 = np.sum((arr > -1) & (arr <= 1))
    
    # Count samples outside the range [-4, 4]
    out_of_range_count = np.sum((arr <= -4) | (arr > 4))
    
    # Return the counts
    return_arr = np.array([count_neg1_pos1, count_neg2_neg1_pos1_pos2, 
                            count_neg3_neg2_pos2_pos3, count_neg4_neg3_pos3_pos4, 
                            out_of_range_count])
    
    total = count_neg1_pos1 + count_neg2_neg1_pos1_pos2 + count_neg3_neg2_pos2_pos3 
    + count_neg4_neg3_pos3_pos4 + out_of_range_count
    
    if (total != len(arr)):
        raise ValueError("Count donsen't contain all the samples.")
    
    return return_arr



