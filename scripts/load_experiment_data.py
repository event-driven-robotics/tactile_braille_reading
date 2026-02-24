import numpy as np

data_path = "./results/20260115_0833_exploration/20260224_080144"
file_name = "braille_reading_rsnn_5_neurons_A_B_rep_1.npz"

data = np.load(f"{data_path}/{file_name}")

print(data.files)