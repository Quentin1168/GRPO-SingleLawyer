import matplotlib.pyplot as plt
import pandas as pd
import wandb

# ==========================================
# 1. Guideline-Compliant Typography & Settings
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,  # Base font size (> 6pt)
    "axes.labelsize": 8.5,  # Axis label font size (> 6pt)
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 1200,  # 1200 DPI requirement
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,  # Embed fonts as TrueType
    "ps.fonttype": 42,
})

# ==========================================
# 2. Smoothing Helper Function (EMA)
# ==========================================
# SMOOTHING_WEIGHT: 0.0 (no smoothing) to 0.99 (very smooth). 0.85 is standard for RL.
SMOOTHING_WEIGHT = 0.85


def smooth_curve(series: pd.Series, weight: float = 0.85) -> pd.Series:
  """Exponential moving average (EMA) smoothing matching W&B dashboard smoothing."""
  return series.ewm(alpha=(1.0 - weight), adjust=False).mean()


# ==========================================
# 3. Authenticate & Connect
# ==========================================
api = wandb.Api()
run = api.run("quentinchang1168-uq/GRPO-SingleLawyer/grpo-lawyer-56")

metrics_to_plot = [
    ("custom/reward_mean", "Mean Total Reward"),
    ("train/milestone_frac", "Mean Milestone Reward"),
    ("train/judge_reward", "Mean Judge Reward"),
]

# Build step lookup table
print("Fetching training step mapping...")
step_records = list(run.scan_history(keys=["_step", "training_step"]))
step_df = (
    pd.DataFrame(step_records)
    .dropna(subset=["training_step"])
    .sort_values("_step")
)

has_training_step = not step_df.empty
x_col = "training_step" if has_training_step else "_step"
x_label = "Training Steps Elapsed" if has_training_step else "Steps Elapsed"

# ==========================================
# 4. Fetch & Smooth Each Metric
# ==========================================
processed_data = []

for raw_key, display_name in metrics_to_plot:
  print(f"Fetching '{raw_key}'...")
  records = list(run.scan_history(keys=["_step", raw_key]))
  df_metric = (
      pd.DataFrame(records).dropna(subset=[raw_key]).sort_values("_step")
  )

  if df_metric.empty:
    print(f"Warning: No valid rows found for {raw_key}")
    continue

  if has_training_step:
    df_metric = pd.merge_asof(
        df_metric, step_df, on="_step", direction="backward"
    )
    df_metric = df_metric.dropna(subset=["training_step"])

  # Calculate smoothed metric
  df_metric["smoothed"] = smooth_curve(
      df_metric[raw_key], weight=SMOOTHING_WEIGHT
  )
  processed_data.append((raw_key, display_name, df_metric))

# ==========================================
# 5. Save Individual Figures
# ==========================================
print("\nSaving individual figures...")
for raw_key, display_name, df_metric in processed_data:
  fig, ax = plt.subplots(figsize=(3.2, 2.3))

  # 1. Faint raw line in background (halftone friendly)
  ax.plot(
      df_metric[x_col],
      df_metric[raw_key],
      color="#000000",
      alpha=0.18,
      linewidth=0.75,
      label="Raw",
  )

  # 2. Solid bold smoothed line
  ax.plot(
      df_metric[x_col],
      df_metric["smoothed"],
      color="#000000",
      linewidth=1.3,
      linestyle="-",
      label="Smoothed",
  )

  ax.set_xlabel(x_label)
  ax.set_ylabel(display_name)
  ax.grid(True, linestyle=":", linewidth=0.5, color="#888888", alpha=0.7)
  ax.set_axisbelow(True)
  ax.locator_params(axis="x", nbins=4)
  ax.locator_params(axis="y", nbins=5)

  plt.tight_layout()

  clean_filename = raw_key.replace("/", "_")
  fig.savefig(f"{clean_filename}.pdf", format="pdf", bbox_inches="tight")
  fig.savefig(f"{clean_filename}.eps", format="eps", bbox_inches="tight")
  fig.savefig(
      f"{clean_filename}.png",
      format="png",
      dpi=1200,
      bbox_inches="tight",
  )
  plt.close(fig)
  print(f"  [✓] Saved {clean_filename} (.pdf, .eps, .png @ 1200 DPI)")

# ==========================================
# 6. Save Combined 1x3 Row Figure
# ==========================================
if processed_data:
  print("\nSaving combined 1x3 row figure...")
  num_plots = len(processed_data)
  fig, axes = plt.subplots(1, num_plots, figsize=(6.8, 2.2))

  if num_plots == 1:
    axes = [axes]

  for ax, (raw_key, display_name, df_metric) in zip(axes, processed_data):
    # Faint raw line
    ax.plot(
        df_metric[x_col],
        df_metric[raw_key],
        color="#000000",
        alpha=0.18,
        linewidth=0.75,
    )

    # Bold smoothed line
    ax.plot(
        df_metric[x_col],
        df_metric["smoothed"],
        color="#000000",
        linewidth=1.3,
        linestyle="-",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(display_name)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#888888", alpha=0.7)
    ax.set_axisbelow(True)
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=5)

  plt.tight_layout()
  fig.savefig("reward_metrics_1x3.pdf", format="pdf", bbox_inches="tight")
  fig.savefig(
      "reward_metrics_1x3.png", format="png", dpi=1200, bbox_inches="tight"
  )
  plt.close(fig)
  print("  [✓] Saved combined figure to reward_metrics_1x3.pdf/.png")