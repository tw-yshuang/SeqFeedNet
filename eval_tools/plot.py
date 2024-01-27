import pandas as pd
import matplotlib.pyplot as plt

# Loading the datasets
file_path_no1eb = 'out/no1eb_Overall.csv'
file_path_final = 'out/final_Overall.csv'

df_no1eb = pd.read_csv(file_path_no1eb)
df_final = pd.read_csv(file_path_final)

# Plotting F_score from both datasets
plt.figure(figsize=(10, 6))
plt.plot(df_final.index, df_final["F_score"], label='SeqFiT')
plt.plot(df_no1eb.index, df_no1eb["F_score"], label='noSeqFiT')
# plt.title("Comparison of SeqFiT and noSeqFiT")
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.legend()
plt.grid(True)
plt.savefig('out/Compare/SeqFiT_prove.png')
