# File: visualization/plot_performance.py
import pandas as pd
import matplotlib.pyplot as plt

# Read performance data from CSV
df = pd.read_csv('../benchmarks/performance.csv')
print(df)

# Plot a bar chart comparing CPU and GPU times
plt.figure(figsize=(6,4))
plt.bar(df['Version'], df['Time_sec'], color=['skyblue', 'salmon'])
plt.ylabel('Execution Time (seconds)')
plt.title('CPU vs. GPU Execution Time')
plt.savefig('performance_comparison.png')
plt.show()
