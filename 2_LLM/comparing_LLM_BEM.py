import pickle
import pandas as pd
import matplotlib.pyplot as plt

bem_data = pd.read_csv("data/BEM_data_1.2.csv")
with open("data/LLM_data.pkl", "rb") as f:
    LLM_data = pickle.load(f)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

n_elems = 6  # assumes (cases, elements)
y = np.array(LLM_data["y_lst"])

alpha = np.array(LLM_data["alpha"][0])
Cl    = np.array(LLM_data["Cl"][0])
Cd    = np.array(LLM_data["Cd"][0])
phi   = np.array(LLM_data["phi"][0])


fig, axs = plt.subplots(2, 2, figsize=(11, 8))


for i in range(n_elems):
    axs[0, 0].plot(y[i*n_elems:(i+1)*n_elems], alpha[i*n_elems:(i+1)*n_elems], label=f"elem {i+1}")

axs[0, 0].set_title("Angle of Attack")
axs[0, 0].grid()
axs[0, 0].legend()



plt.tight_layout()
plt.show()