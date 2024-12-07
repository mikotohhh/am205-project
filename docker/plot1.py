import os
import json
import matplotlib.pyplot as plt

# Define the folder containing the JSON files
results_folder = "results"

# Data storage
data = {}

# Iterate through the JSON files
for filename in os.listdir(results_folder):
    if filename.endswith(".json"):
        # Extract method and timeout from the filename
        parts = filename.split("_")
        method = parts[2]
        timeout = float(parts[-1].replace("s.json", "").replace("timeout_", ""))
        
        # Load the JSON file content
        filepath = os.path.join(results_folder, filename)
        with open(filepath, 'r') as file:
            content = json.load(file)
            solved_ratio = content["overall_metrics"]["solved_ratio"]
            solved_count, total_problems = map(int, solved_ratio.split("/"))
            solved_ratio_value = solved_count / total_problems
        
        # Store the data
        if method not in data:
            data[method] = []
        data[method].append((timeout, solved_ratio_value))

# Prepare data for plotting
plt.figure(figsize=(8, 6))

for method, values in data.items():
    values.sort(key=lambda x: x[0])  # Sort by timeout
    timeouts, ratios = zip(*values)
    plt.plot(timeouts, ratios, marker='o', label=method)

# Customize the plot
plt.title("Problems Solved Ratio vs Timeout")
plt.xlabel("Timeout (t)")
plt.ylabel("P p:r(p,s) <= t")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig("plot1.jpg")

# Print a confirmation message
print("Plot saved as 'plot1.jpg'.")
