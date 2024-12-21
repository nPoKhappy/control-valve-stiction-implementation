"""combine r_value data"""

import os
import scipy.io
from new_project.thesis_2.sigmoid_class import Sigmoid
from PIL import Image


def create_gif(chemical_data, loop_list, gif_name):
    image_files = []  # List to store image paths
    for loop_name in loop_list:
        # Access the specific loop
        loop = chemical_data[f'loop{loop_name}'][0, 0]
        pv = loop['PV']  # shape (1000, )
        op = loop['OP']  # shape (1000, )
        
        # Generate the plot and save as image
        img_filename = f"loop_{loop_name}.png"
        sigmoid = Sigmoid(op=op, pv=pv)
        sigmoid.delta_pv_op_plot(name=str(loop_name), filename=img_filename)
        image_files.append(img_filename)

    # Open images and create a GIF
    images = [Image.open(img) for img in image_files]
    images[0].save(
        gif_name,
        save_all=True,
        append_images=images[1:],
        duration=1500,  # Duration for each frame (in milliseconds)
        loop=0  # 0 for infinite loop
    )
    
    print(f"GIF created successfully: {gif_name}")


"""import data from idsb"""
# Load the .mat file
mat_contents = scipy.io.loadmat('new_project/isdb10.mat')

# Access the 'cdata' structure within 'mat_contents'
cdata = mat_contents['cdata'][0,0]

# Access the 'chemicals' structure within 'cdata'
chemicals = cdata['chemicals'][0, 0]





# Example usage:
loop_list = [3, 13, 12, 11, 2, 23, 32, 6, 14, 1, 10, 29, 24]
gif_name = 'chemical_loops.gif'

# Call the function to generate the GIF
create_gif(chemicals, loop_list, gif_name)

