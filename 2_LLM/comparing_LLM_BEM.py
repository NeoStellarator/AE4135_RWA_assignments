import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
bem_data = pd.read_csv("data/BEM_data_1.2.csv")
with open("data/LLM_data.pkl", "rb") as f:
    LLM_data = pickle.load(f)

n_elem = 40
for i in range(6):
    start = i*n_elem
    end = (i+1)*n_elem
    plt.plot(LLM_data["y"][start:end],LLM_data["Cl"][start:end],label = f"blade {i}")
plt.ylabel("Cl")
plt.xlabel("y[m]")
plt.legend()
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(bem_data["r_R"]*0.7, bem_data["alpha"])
axes[0].plot(LLM_data["y"][0:n_elem], LLM_data["alpha"][0:n_elem])
axes[0].set_title("Alpha")

axes[1].plot(bem_data["r_R"]*0.7, bem_data["phi"])
axes[1].plot(LLM_data["y"][0:n_elem], LLM_data["phi"][0:n_elem])
axes[1].set_title("Phi")

axes[2].plot(bem_data["r_R"]*0.7, bem_data["Cl"])
axes[2].plot(LLM_data["y"][0:n_elem], LLM_data["Cl"][0:n_elem])
axes[2].set_title("Cl")

for ax in axes:
    ax.legend(["BEM", "LLM"])
    ax.set_xlabel("r/R")

plt.tight_layout()
plt.show()