import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 1. Load the text file (space/whitespace delimited)
file_path = "log.txt"  # Replace with your actual file path



# Read data, treating any sequence of whitespace as column separators
df = pd.read_csv(file_path, sep=r"\s+")

# Ensure column names are clean of extraneous spaces
df.columns = df.columns.str.strip()

# If your 'Grad_Norm' column header was truncated as 'Grad_No', rename it:
df = df.rename(columns={df.columns[-1]: "Grad_Norm"})

# 2. Group by 'Global_Step' and average the 3 records per step
df_avg = df.groupby("Global_Step", as_index=False).mean()

def clean_rl_spikes(series, high_quantile=0.92, window_size=15):
    """Aggressively removes RL spikes using two steps:

    1. Removes extreme values strictly above a high percentile (e.g. 92nd).
    2. Applies a rolling median filter to erase multi-step residual spikes.
    """
    cleaned = series.copy()

    # Step 1: Replace values exceeding the upper percentile threshold with NaN
    upper_threshold = cleaned.quantile(high_quantile)
    cleaned[cleaned > upper_threshold] = np.nan

    # Step 2: Fill NaNs via linear interpolation so line stays connected
    cleaned = cleaned.interpolate(method="linear").bfill().ffill()

    # Step 3: Rolling Median to smooth out remaining multi-step mini-spikes
    cleaned_smooth = (
        cleaned.rolling(window=window_size, center=True, min_periods=1)
        .median()
        .ewm(span=3)
        .mean()
    )

    return cleaned_smooth

df_avg["Loss_Clean"] = clean_rl_spikes(df_avg["Loss"])
df_avg["Reward_Clean"] = clean_rl_spikes(df_avg["Mean_Reward"])
df_avg["Grad_Norm_Clean"] = clean_rl_spikes(df_avg["Grad_Norm"])

SMOOTHING_SPAN = 5



# 3. Create 3 Subplots in One Figure
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)



# Plot 1: Loss
axes[0].plot(
    df_avg["Global_Step"],
    df_avg["Loss_Clean"],
    color="#d62728",
    linewidth=1.8,
    marker="o",
    markersize=3,
)
axes[0].set_ylabel("Loss")
axes[0].set_title("Average Loss per Global Step")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Plot 2: Mean Reward
axes[1].plot(
    df_avg["Global_Step"],
    df_avg["Reward_Clean"],
    color="#2ca02c",
    linewidth=1.8,
    marker="o",
    markersize=3,
)
axes[1].set_ylabel("Mean Reward")
axes[1].set_title("Average Mean Reward per Global Step")
axes[1].grid(True, linestyle="--", alpha=0.5)

# Plot 3: Grad Norm
axes[2].plot(
    df_avg["Global_Step"],
    df_avg["Grad_Norm_Clean"],
    color="#1f77b4",
    linewidth=1.8,
    marker="o",
    markersize=3,
)
axes[2].set_xlabel("Global Step")
axes[2].set_ylabel("Grad Norm")
axes[2].set_title("Average Grad Norm per Global Step")
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("metrics_summary.png", dpi=300)
plt.show()