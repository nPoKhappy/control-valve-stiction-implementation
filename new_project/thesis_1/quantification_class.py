import numpy as np
import matplotlib.pyplot as plt



class Quantification():
    def __init__(self, S_expand: np.ndarray, op: np.ndarray) -> None:
        self.s_expand = S_expand
        self.op = op
    
    def perfrom(self)-> None:
        stiction_bands = []
        start_idx = None  # Initialize start_idx

        # Iterate through S_expand to detect stiction bands
        for i in range(1, len(self.s_expand)):
            if self.s_expand[i] == 1 and self.s_expand[i - 1] == 0:  # Detect valve begin to have stiction (0 to 1)
                start_idx = i
            elif self.s_expand[i] == 0 and self.s_expand[i - 1] == 1:  # Detect valve begins to not have stiction (1 to 0)
                if start_idx is not None:  # Ensure there's a valid start index
                    end_idx = i - 1
                    stiction_band = abs(self.op[start_idx] - self.op[end_idx])
                    stiction_bands.append(stiction_band)
                start_idx = None  # Reset start_idx

        # Handle the case where the sequence starts with stiction (S_expand[0] == 1)
        if self.s_expand[0] == 1:
            start_idx = 0
            # Find the next transition from 1 to 0
            for i in range(1, len(self.s_expand)):
                if self.s_expand[i] == 0:
                    end_idx = i - 1
                    stiction_band = abs(self.op[start_idx] - self.op[end_idx])
                    stiction_bands.append(stiction_band)
                    break

        
        # Convert to numpy array
        stiction_bands = np.array(stiction_bands)

        # Calculate the 95% band
        percentile_95 = np.percentile(stiction_bands, 95)

        print(f"percentile95:{percentile_95:.3f}")

        # Plot the stiction band histogram
        plt.hist(stiction_bands, bins=4, color='c',edgecolor='black')

        # Label the 95% stiction band
        plt.axvline(percentile_95, color='red', linewidth=2)
        # Text "95th percentile"
        plt.text(percentile_95 - 1,  42, 
                '95th percentile', color='black')
        # Arrow
        plt.annotate('',
            #where the tick point at
             xy=(percentile_95, 35), 
            # The tail 
             xytext=(percentile_95 - 0.5, 40),
             arrowprops=dict(facecolor='black', shrink=0.05))


        # Set the title and label
        plt.title('Distribution of Stiction Bands (η)')
        plt.xlabel('Stiction Band (η)')
        plt.ylabel('Frequency')

        # Show the plot
        plt.show()