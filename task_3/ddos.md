DDoS Attack Analysis Report

Overview

  This report analyzes the web server log file s_chalauri25_18943_server.log to identify time intervals of a DDoS attack using regression analysis. 
The approach includes parsing timestamps, aggregating request counts per minute, fitting a linear regression model to establish the baseline traffic trend, calculating residuals, 
detecting anomalies (residuals > 3× standard deviation), and grouping consecutive anomalies into attack intervals. The provided visualization confirms a dramatic spike in traffic consistent with a DDoS event.

Log File

  The event log file is available at: s_chalauri25_18943_server.log

Detected DDoS Time Interval(s)

  •	2024-03-22 18:55:00+04:00 to 2024-03-22 18:57:00+04:00

  This interval contains the extreme traffic spike (peak ~11,564 requests/minute) detected as statistical anomalies by the regression model.

Methodology

    1.	Parse Log: Extract timestamps using regex and datetime.
    2.	Aggregate: Group by minute to obtain requests-per-minute time series.
    3.	Regression: Linear regression (time in minutes as predictor, requests as response) using statsmodels.OLS.
    4.	Anomaly Detection: Residuals exceeding 3σ threshold.
    5.	Interval Construction: Merge consecutive anomalous minutes.
    6.	Visualization: Plot actual counts, regression line, and anomalies.

Full script: ddos_analysis.py uploaded in directory

Results and Visualizations

Key Metrics at Peak:

    Minute	Requests	Predicted	Residual	Anomaly
    2024-03-22 18:55:00+04:00	~9408	~2005	+7403	True
    2024-03-22 18:56:00+04:00	~11564	~2035	+9529	True
    
Visualization: Request Counts with Regression and Anomalies

<img width="1200" height="600" alt="ddos_visualization" src="https://github.com/user-attachments/assets/db029eba-d2cb-4277-ba51-7c786c62adbc" />

The plot clearly shows normal traffic fluctuating around the upward regression trend until a massive spike at approximately 18:55, marked by orange anomaly points. 
This deviation is far beyond normal variation and matches the classic signature of a DDoS attack.

