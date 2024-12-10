import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV data
data = pd.read_csv('results.csv')

# Create a unique problem identifier combining Problem and Dimension
data['ProblemID'] = data['Problem'] + '_' + data['Dimension'].astype(str)

# Get unique methods
methods = sorted(data['Method'].unique())
n_problems = len(data['ProblemID'].unique())
print(n_problems)

# Create time points for plotting
max_time = data['Time (s)'].max()
print(max_time)
t_points = np.linspace(0, max_time+50, 100000)

# Initialize profiles dictionary
profiles = {}

# Calculate profiles for each method
for method in methods:
    method_data = data[data['Method'] == method].copy()
    solved_times = method_data[method_data['Success']]['Time (s)'].values
    
    # Calculate percentage of problems solved at each time point
    profile = [np.sum(solved_times <= t) / n_problems for t in t_points]
    profiles[method] = profile

# Plotting
plt.figure(figsize=(10, 6))
line_styles = ['-', '--', '-.', ':', '-']
colors = ['b', 'g', 'r', 'brown', 'm']

for (method, profile), ls, color in zip(profiles.items(), line_styles, colors):
    plt.plot(t_points, profile, ls, label=method, color=color, alpha=0.6)

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Fraction of problems solved')
plt.xlim(0,1750)
plt.ylim(0, 1.1)
plt.legend()
plt.title('Performance Profile in High Dimension Problems')
plt.tight_layout()
plt.show()