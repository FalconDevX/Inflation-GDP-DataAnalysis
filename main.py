import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
inflation_df = pd.read_csv("yearly_avg_inflation.csv")
inflation_df['procentowa_zmiana_inflacji'] = inflation_df['Inflation'].pct_change() * 100

debt_df = pd.read_csv("debt.csv")
debt_change_df = pd.read_csv("debt_change.csv")
ppp_df = pd.read_csv("ppp.csv")

merged_df = pd.merge(inflation_df, debt_df, on=['GeoCode', 'Country', 'Year'], how='inner')


merged_df = pd.merge(merged_df, debt_change_df, on=['GeoCode', 'Country', 'Year'], how='inner')

# merged_df = pd.merge(merged_df, ppp_df, on=['GeoCode', 'Country', 'Year'], how='inner')

print(merged_df.head(40))


# Filter for Poland
poland_df = merged_df[merged_df['Country'] == 'Belgium']

# Drop non-numeric columns (e.g., GeoCode, Country, Year)
numeric_df = poland_df.select_dtypes(include='number')

corr_matrix = numeric_df.corr()

print(poland_df.head(100))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Poland')
plt.show()

# Usunięcie danych dla zbiorczych indeksów, takich jak EA (strefa euro) itp.
excluded_codes = ['EA', 'EU', 'EU27_2020', 'EU28', 'EA20', 'EA19', 'EU19', 'EU28_2020']
filtered_df = merged_df[~merged_df['GeoCode'].isin(excluded_codes)]

# Ponowne obliczenie korelacji dla wszystkich krajów (bez EA, EU itp.)
numeric_df_filtered = filtered_df.select_dtypes(include='number')
corr_matrix_filtered = numeric_df_filtered.corr()

# Rysowanie wykresu
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_filtered, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for All Countries (excluding EA/EU aggregates)')
plt.tight_layout()
plt.show()