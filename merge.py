import pandas as pd

# Wczytaj dane z plików CSV
inflation_df = pd.read_csv('yearly_avg_inflation.csv')
debt_df = pd.read_csv('debt.csv')
debt_change_df = pd.read_csv('debt_change.csv')
ppp_df = pd.read_csv('ppp.csv')

# Połącz wszystkie trzy DataFrames
merged_df = pd.merge(
    inflation_df,
    debt_df,
    on=['GeoCode', 'Country', 'Year'],
    how='outer'
)
merged_df = pd.merge(
    merged_df,
    debt_change_df,
    on=['GeoCode', 'Country', 'Year'],
    how='outer'
)
merged_df = pd.merge(
    merged_df,
    ppp_df,
    on=['GeoCode', 'Country', 'Year'],
    how='outer'
)

# Zapisz wynik do nowego pliku CSV
merged_df.to_csv('merged_data.csv', index=False)