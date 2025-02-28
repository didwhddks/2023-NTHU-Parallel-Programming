import matplotlib.pyplot as plt

# Sample data: Execution times (speedup over CPU)
labels = ['CPU', 'GPU Baseline', 'GPU Coalesced', 'GPU Shared', 'GPU Optimal']
speedup = [1, 16.1, 44.1, 39.3, 61.5]
colors = ['gray', 'royalblue', 'limegreen', 'crimson', 'darkorange']

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 5))

# Create bar plot with rounded edges
bars = ax.bar(labels, speedup, color=colors, edgecolor='black', linewidth=1.2)

# Labels and title
ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
ax.set_title('GPU vs CPU Speedup Comparison', fontsize=16, fontweight='bold', pad=15)

# Show values on top of bars
for bar, v in zip(bars, speedup):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold', color='black')

# Add grid lines for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.6)

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the figure
plt.savefig('result.png', dpi=300, bbox_inches='tight')
