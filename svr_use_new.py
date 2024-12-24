"""train model"""
from new_project.final.svr_class_new import Svr

file_path : str = "new_project/CVOP_20220201~0430.xlsx" # The excel file name

# list inside with list inside with str ["PV", "CO"]
# Each one has 128160 samples
data_cols : list[list[str]] = [["FT_107", "FV_107.OUT"], ["FT_117", "FV_117.OUT"], ["FT_174", "FV_174.OUT"], ["FT_181", "FV_181.OUT"],
             ["FT_2004", "FV_2004.OUT"], ["FT_306", "FV_306.OUT"], ["FT_308", "FV_308.OUT"]] # 

print(f"There are {len(data_cols)} control loop data.")

svr : Svr = Svr(file_name = file_path, 
          data_col = data_cols, 
          c_start = 1, c_end = 1.05,
          epsilon_start = 0.1, epsilon_end = 0.11)

svr.train_valid(store = 1, store_path = "new_project/csv/final/svr_model/CVOP_20220201~0430.pkl")
# svr.draw_validation_mse(show = 1)
svr.draw_r_val()