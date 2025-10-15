import pandas as pd

# The name of your original large file
original_file = 'AIS_2024_12_311.csv'

# The name for your new, smaller file
new_file = 'AIS_data_sampled.csv'

# The number of random rows you want in your new file
num_rows_to_sample = 20000

print(f"Reading the original file: {original_file}...")
# Note: This might take a minute depending on your computer's memory
df = pd.read_csv(original_file)

print(f"Taking a random sample of {num_rows_to_sample} rows...")
sampled_df = df.sample(n=num_rows_to_sample)

print(f"Saving the smaller file as: {new_file}...")
# index=False is important to avoid adding an extra column
sampled_df.to_csv(new_file, index=False)

print("✅ Done! Your new, smaller data file is ready.")