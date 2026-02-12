import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Read the log file
with open('s_chalauri25_18943_server.log', 'r') as f:
    lines = f.readlines()

# Parse timestamps
timestamps = []
for line in lines:
    if line.strip():
        start = line.find('[')
        end = line.find(']')
        if start != -1 and end != -1:
            ts_str = line[start+1:end]
            try:
                dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S%z')
                timestamps.append(dt)
            except ValueError:
                pass

# Create DataFrame
df = pd.DataFrame({'timestamp': timestamps})
df = df.sort_values('timestamp')
df['minute'] = df['timestamp'].dt.floor('min')
counts = df.groupby('minute').size().reset_index(name='requests')



# Convert time to numeric
start_time = counts['minute'].min()
counts['time_min'] = (counts['minute'] - start_time).dt.total_seconds() / 60

# Linear regression
X = sm.add_constant(counts['time_min'])
y = counts['requests']
model = sm.OLS(y, X).fit()
counts['predicted'] = model.predict(X)
counts['residual'] = y - counts['predicted']

# Detect anomalies
residual_std = counts['residual'].std()
threshold = 3 * residual_std
counts['anomaly'] = np.abs(counts['residual']) > threshold

# Find intervals
anomalies = counts[counts['anomaly']].sort_values('minute')
if not anomalies.empty:
    intervals = []
    current_start = anomalies.iloc[0]['minute']
    current_end = current_start
    for i in range(1, len(anomalies)):
        if anomalies.iloc[i]['minute'] == current_end + pd.Timedelta(minutes=1):
            current_end = anomalies.iloc[i]['minute']
        else:
            intervals.append((current_start, current_end + pd.Timedelta(minutes=1)))
            current_start = anomalies.iloc[i]['minute']
            current_end = current_start
    intervals.append((current_start, current_end + pd.Timedelta(minutes=1)))



plt.figure(figsize=(12, 6))
plt.plot(counts['minute'], counts['requests'], label='Requests per Minute')
plt.plot(counts['minute'], counts['predicted'], label='Regression Line', color='red')
plt.scatter(counts[counts['anomaly']]['minute'], counts[counts['anomaly']]['requests'], color='orange', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Number of Requests')
plt.title('Request Counts with Regression and Anomalies')
plt.legend()
plt.grid(True)
plt.savefig('ddos_visualization.png')  # Save the plot
plt.show()