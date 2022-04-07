import matplotlib.pyplot as plt
import numpy as np

def plot_results(true_data, true_quantiles, estimated_quantiles, quantile):
    x = np.arange(0, len(true_data))
    plt.plot(x, true_data, label="Data", linestyle="-")
    plt.plot(x, true_quantiles, label=f"True ({quantile}) quantiles", linestyle="--")
    plt.plot(x, estimated_quantiles, label=f"Est. ({quantile}) quantiles", linestyle="-.")
    plt.legend()
    plt.show()
    plt.close()

