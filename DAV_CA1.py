import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# xls = pd.ExcelFile("YALE-EPI.xlsx")

# df = pd.read_excel(xls, sheet_name="Data")

# # Pivot the dataframe to have the indicators as columns
# df_pivot = df.pivot_table(index="Economy Name", columns="Indicator", values="2022")

# # Clean of the column names in the pivoted dataframe for readability
# df_pivot.columns = df_pivot.columns.str.replace(
#     "Environmental Performance Index: ", "", regex=False
# )
    
# # Defining the columns to keep based on the actual column names in the dataset
# columns_to_keep = [
#     "PM2.5 exposure",
#     "SO2 exposure",  
#     "Household solid fuels",
#     "Unsafe drinking water",
#     "Unsafe sanitation",
#     "Lead exposure",
#     "Biodiversity Habitat Index",
#     "Terrestrial biome protection (global weights)",
#     "Tree cover loss",
#     "Ocean Plastics",
#     "Greenhouse gas emissions per capita",
# ]

# # Creating a new dataframe with only the columns to keep
# available_columns = [col for col in columns_to_keep if col in df_pivot.columns]
# df_cleaned = df_pivot[available_columns]

# # Saving the cleaned dataframe to an excel file
# df_cleaned.to_excel("EPI_Cleaned.xlsx")


df_cleaned = pd.read_excel("EPI_Cleaned.xlsx", index_col=0)

# Here I'm checking for missing values
print(df_cleaned.isnull().sum())





