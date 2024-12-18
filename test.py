import pandas as pd

# Load Excel file
file_path = "new_project/CVOP_20220201~0430.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path, header = 1, engine="openpyxl") # Header set to 1 for header in the second row and engine 
                                                             # is to make sure compatibility with .xlsx files.
                                                             # need to install openpyxl
# Access specific columns
ft_107 = df["FT_107"]          # Access the 'FT_107' column
fv_107_out = df["FV_107.OUT"]    # Access the 'FV_107.O' column

# Print or work with the data
print("FT_107 Data:")
print(ft_107.head())

print("\nFV_107.O Data:")
print(fv_107_out.head())
