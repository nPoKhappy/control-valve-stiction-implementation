"""train model"""
from new_project.final.svr_class_new import Svr

file_name = "CVOP_20241001~1209.xlsx" # The excel file name

c_start, c_end, epsilon_start, epsilon_end = (1, 1.05, 0.1, 0.11)

# list inside with list inside with str ["PV", "CO"]
# Each one has 128160 samples
data_cols : list[tuple[str]] = [
    ("FT_107", "FV_107.OUT"), 
    ("FT_117", "FV_117.OUT"), 
    ("FT_174", "FV_174.OUT"), 
    ("FT_181", "FV_181.OUT"), 
    ("FT_2004", "FV_2004.OUT"), 
    ("FT_306", "FV_306.OUT"), 
    ("FT_308", "FV_308.OUT")
]

print(f"There are {len(data_cols)} control loop data.")


file_path : str = f"new_project/{file_name}"
svr : Svr = Svr(file_name = file_path, 
          data_col = data_cols, 
          c_start = c_start, c_end = c_end,
          epsilon_start = epsilon_start, epsilon_end = epsilon_end)

svr.train_valid(store = 1, store_path = f"new_project/csv/final/svr_model/{file_name}.pkl")
# svr.draw_validation_mse(show = 1)
svr.draw_r_val(file_name = file_name)