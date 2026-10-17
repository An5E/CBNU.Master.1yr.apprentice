import seaborn as sns
import matplotlib.pyplot as plt

cols = [
    "HUFL", "HULL", "MUFL",
    "MULL", "LUFL", "LULL", "OT"
]

corr = train_df[cols].corr(method="pearson")

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("ETTm2 Feature Correlation")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)