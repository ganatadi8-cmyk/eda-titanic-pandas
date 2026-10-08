import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set overall design aesthetics & style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold'
})

# Create directory to save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Load & Preprocess Dataset
print("Loading Titanic Dataset...")
df = sns.load_dataset('titanic')

# Data cleaning
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['fare'] = df['fare'].fillna(df['fare'].median())

print("Dataset loaded successfully. Generating Data Visualizations...\n")

# ---------------------------------------------------------
# CHART 1: Bar Chart - Survival Rate by Passenger Class & Gender
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
bar_ax = sns.barplot(
    data=df,
    x='pclass',
    y='survived',
    hue='sex',
    palette='Set2',
    ci=None,
    edgecolor='black'
)
plt.title('Survival Rate by Passenger Class and Gender')
plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
plt.ylabel('Survival Rate')
plt.ylim(0, 1.05)
plt.legend(title='Gender', frameon=True)
plt.tight_layout()
bar_path = os.path.join(output_dir, '1_bar_chart_survival.png')
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"[✓] Saved Bar Chart to {bar_path}")

# ---------------------------------------------------------
# CHART 2: Scatter Plot - Age vs Fare by Survival Status
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))
scatter_ax = sns.scatterplot(
    data=df,
    x='age',
    y='fare',
    hue='survived',
    style='survived',
    palette={0: '#e74c3c', 1: '#2ecc71'},
    alpha=0.8,
    s=70,
    edgecolor='w'
)
plt.title('Scatter Plot: Age vs Fare (Colored by Survival Status)')
plt.xlabel('Age (Years)')
plt.ylabel('Fare ($)')
plt.yscale('log')  # Log scale for better fare visualization
plt.legend(title='Survived', labels=['No (0)', 'Yes (1)'], frameon=True)
plt.tight_layout()
scatter_path = os.path.join(output_dir, '2_scatter_plot_age_fare.png')
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"[✓] Saved Scatter Plot to {scatter_path}")

# ---------------------------------------------------------
# CHART 3: Histogram & KDE - Age Distribution by Survival
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x='age',
    hue='survived',
    kde=True,
    element='step',
    palette={0: '#34495e', 1: '#1abc9c'},
    bins=30
)
plt.title('Age Distribution and Kernel Density Estimate (KDE)')
plt.xlabel('Age')
plt.ylabel('Passenger Count')
plt.legend(title='Survived', labels=['Yes (1)', 'No (0)'], frameon=True)
plt.tight_layout()
hist_path = os.path.join(output_dir, '3_histogram_age_distribution.png')
plt.savefig(hist_path, dpi=300)
plt.close()
print(f"[✓] Saved Histogram & KDE plot to {hist_path}")

# ---------------------------------------------------------
# CHART 4: Box Plot - Fare Distribution across Passenger Classes
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x='pclass',
    y='fare',
    palette='Pastel1',
    showfliers=True,
    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5}
)
plt.title('Box Plot: Fare Distribution Across Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Fare ($)')
plt.ylim(0, 300)  # Zoom in to see quantiles clearly
plt.tight_layout()
box_path = os.path.join(output_dir, '4_box_plot_fare_pclass.png')
plt.savefig(box_path, dpi=300)
plt.close()
print(f"[✓] Saved Box Plot to {box_path}")

# ---------------------------------------------------------
# CHART 5: Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'alone']
corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Correlation Heatmap Matrix')
plt.tight_layout()
heatmap_path = os.path.join(output_dir, '5_correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"[✓] Saved Correlation Heatmap to {heatmap_path}")

# ---------------------------------------------------------
# CHART 6: Comprehensive 2x3 Dashboard Grid
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Titanic Dataset: Data Visualization Dashboard', fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: Bar Chart
sns.barplot(data=df, x='pclass', y='survived', hue='sex', palette='Set2', ci=None, ax=axes[0, 0])
axes[0, 0].set_title('Survival Rate by Class & Sex')

# Subplot 2: Scatter Plot
sns.scatterplot(data=df, x='age', y='fare', hue='survived', palette={0: '#e74c3c', 1: '#2ecc71'}, alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('Age vs Fare (Log Scale)')

# Subplot 3: Histogram
sns.histplot(data=df, x='age', hue='survived', kde=True, palette={0: '#34495e', 1: '#1abc9c'}, bins=25, ax=axes[0, 2])
axes[0, 2].set_title('Age Distribution by Survival')

# Subplot 4: Box Plot
sns.boxplot(data=df, x='pclass', y='fare', palette='Pastel1', ax=axes[1, 0])
axes[1, 0].set_ylim(0, 300)
axes[1, 0].set_title('Fare Outliers & Quantiles by Class')

# Subplot 5: Heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('Correlation Heatmap')

# Subplot 6: Countplot - Embark Town Survival
sns.countplot(data=df, x='embark_town', hue='survived', palette='Accent', ax=axes[1, 2])
axes[1, 2].set_title('Passenger Count by Embark Town & Survival')

plt.tight_layout(rect=[0, 0, 1, 0.95])
dashboard_path = os.path.join(output_dir, '6_combined_dashboard.png')
plt.savefig(dashboard_path, dpi=300)
plt.close()
print(f"[✓] Saved Complete Dashboard to {dashboard_path}")

print("\nAll visualizations generated and saved successfully!")
