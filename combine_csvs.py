import pandas as pd
df1 = pd.read_csv('results/final_run_results.csv')
df2 = pd.read_csv('results/climate_run_results.csv')
df3 = pd.read_csv('results/nli_run_results.csv')
combined_df = pd.concat([df1, df2, df3])
combined_df.to_csv('results/combined_results.csv', index=False)
