from new_project.final.svr_class_new import Svr



file_name = "" # The excel file name

data_cols = [[], [], [], []] # 

svr = Svr(file_name = file_name, 
          data_col = data_cols, 
          c_start = 0.01, c_end = 1.5,
          epsilon_start = 0.0, epsilon_end = 0.3)

svr.train_valid(store = 0)
svr.draw_validation_mse(show = 1)
svr.draw_r_val()