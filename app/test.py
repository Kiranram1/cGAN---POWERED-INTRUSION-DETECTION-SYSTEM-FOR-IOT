import pandas as pd

# Load the CSV
df = pd.read_csv("/home/kiranram/Downloads/synthetic_data(1).csv")

# Print all column names for reference
print("Columns in your file:")
print(list(df.columns))

# Find protocol and service columns
proto_cols = [col for col in df.columns if col.startswith("proto_")]
service_cols = [col for col in df.columns if col.startswith("service_")]

# Create 'proto' and 'service' columns from one-hot encoding
df['proto'] = df[proto_cols].idxmax(axis=1).str.replace("proto_", "")
df['service'] = df[service_cols].idxmax(axis=1).str.replace("service_", "")

# Now you can do your analysis as before
attack_col = "Attack_type"

print("\nNumber of samples per Attack_type:")
print(df[attack_col].value_counts())

print(f"\n{attack_col} vs proto:")
print(pd.crosstab(df[attack_col], df['proto']))

print(f"\n{attack_col} vs service:")
print(pd.crosstab(df[attack_col], df['service']))
