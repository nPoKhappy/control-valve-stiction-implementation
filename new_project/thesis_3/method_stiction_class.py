import numpy as np
import matplotlib.pyplot as plt




def stiction_signal_plot(pv: np.ndarray, S_expand: np.ndarray, process_name: str) -> None:
    """
    Show the stiction signal on the process variable plot.
    Parameter:
    pv: process varibale data.
    S_expand: stiction signal size where match the process variable.
    Return:
    None.
    """
    # Plotting the Process Variable (PV)
    fig, ax1 = plt.subplots()

    # PV
    ax1.plot(range(pv.size), pv, label='Process variable')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Process variable', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plotting the Stiction Signal (S)
    ax2 = ax1.twinx()
    ax2.plot(range(S_expand.shape[0]), S_expand, 'r-', label='Stiction signal')
    ax2.set_ylabel('Stiction signal', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Adding title and formatting
    plt.title(f'Stiction detection via each Method in {process_name}')
    fig.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
