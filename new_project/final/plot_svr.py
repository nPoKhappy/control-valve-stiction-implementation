import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# r_val = np.loadtxt('new_project/csv/final/svr_model/r_val_svr.csv', delimiter=',')
r_val_valid = np.loadtxt('new_project/csv/final/svr_model/r_val_svr_valid.csv', delimiter=',')
r_val_test = np.loadtxt('new_project/csv/final/svr_model/r_val_svr_test.csv', delimiter=',')

print(f"shape of r_val_test : {r_val_test.shape}\n")
print(f"shape of r_val_valid : {r_val_valid.shape}\n")

# Filter for those data diff larger than threshold
threshold = 0.2
diff = abs(r_val_test[:, 0] - r_val_test[:, 1])
mask = diff <= threshold
r_val_test_filter = r_val_test[mask]

print(f"shape of r_val_test_filter : {r_val_test_filter.shape}\n")

threshold = 0.2
diff = abs(r_val_valid[:, 0] - r_val_valid[:, 1])
mask = diff <= threshold
r_val_valid_filter = r_val_valid[mask]

print(f"shape of r_val_valid_filter : {r_val_valid_filter.shape}\n")

r_val_con = np.concatenate([r_val_test_filter, r_val_valid_filter])
# Sort array
sorted_indices = np.argsort(r_val_con[:, 0])
r_val_con_sorted = r_val_con[sorted_indices]


# accuracy valid : 0.65  test : 0.43

plt.figure(dpi=150)
plt.plot(range(len(r_val_con_sorted[:, 0])), r_val_con_sorted[:, 0] , label="$real$")
plt.plot(range(len(r_val_con_sorted[:, 0])), r_val_con_sorted[:, 1], label="$pred$")
plt.xlabel("$windows$")
plt.ylabel("$R$")
plt.legend()
plt.show()



# # Sample data 
# x_data = np.arange(len(r_val_con_sorted[:, 0]))
# real_r_val = r_val_con_sorted[:, 0]
# pred_r_val = r_val_con_sorted[:, 1]

# # Create a figure and axis
# fig, ax = plt.subplots(dpi=150)
# ax.set_xlabel("$windows$")
# ax.set_ylabel("$R$")
# ax.set_title("Testing for constant stiction control valve.")

# # Initialize the plot elements for real and predicted values
# real_line, = ax.plot([], [], label="$real$")
# pred_line, = ax.plot([], [], linestyle="-", label="$pred$")
# ax.legend()

# # Set the axis limits
# ax.set_xlim(0, len(x_data))
# ax.set_ylim(min(real_r_val.min(), pred_r_val.min()), max(real_r_val.max(), pred_r_val.max()))

# # Update function for the animation
# def update(frame):
#     # Gradually plot the points up to the current frame
#     real_line.set_data(x_data[:frame], real_r_val[:frame])
#     pred_line.set_data(x_data[:frame], pred_r_val[:frame])
#     return real_line, pred_line

# # Create the animation: gradually animate over the number of frames
# ani = FuncAnimation(fig, update, frames=len(x_data), interval=1000, blit=True)

# # Save the animation as a GIF
# ani.save('r_val_animation.gif', writer='pillow')

# # Show the animated plot
# plt.show()
