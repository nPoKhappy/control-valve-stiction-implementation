from new_project.final.svr_class import Svr
from PIL import Image

loop_list = [1, 2, 3, 6, 10, 11, 12, 13, 23, 24, 29, 32]
svr = Svr(loop_list = loop_list, 
          c_start = 0.01, c_end = 1.5,
          epsilon_start = 0.0, epsilon_end = 0.3)
svr.train_valid(store = 0)
svr.draw_validation_mse(show = 1)
svr.draw_r_val()


# def create_gif(gif_name: str, image_files: list):
#     # Open images and create a GIF
#     images = [Image.open(img) for img in image_files]
#     images[0].save(
#         gif_name,
#         save_all=True,
#         append_images=images[1:],
#         duration=1500,  # Duration for each frame (in milliseconds)
#         loop=0  # 0 for infinite loop
#     )
    
#     print(f"GIF created successfully: {gif_name}")

# image_file_list = ["c_0.1_and_epsilon_0.1.png", "c_0.5_and_epsilon_0.05.png", "c_1.1_and_epsilon_0.01.png"]

# create_gif(gif_name = "r_value.gif", image_files = image_file_list)