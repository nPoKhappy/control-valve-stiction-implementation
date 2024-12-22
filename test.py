import pandas as pd
from new_project.thesis_1.plot import plot_pv_sp_op
# Load Excel file
file_path = "new_project/CVOP_20220201~0430.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path, header = 1, engine="openpyxl") # Header set to 1 for header in the second row and engine 
                                                             # is to make sure compatibility with .xlsx files.
                                                             # need to install openpyxl
# Access specific columns
pv = df["FT_306"]          # Access the 'FT_107' column
op = df["FV_306.OUT"]    # Access the 'FV_107.O' column


plot_pv_sp_op(pv = pv, sp = None, co = op, loop_name = file_path)