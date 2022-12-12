#%%
import pandas as pd
import missingno as msno
#%%

# df = pd.read_csv("airbnb-property-listings/tabular_data/listing.csv")
# df_copy = df.copy()
# df_copy = df_copy.replace(r'\r+|\\n+|\t+','', regex=True)


def remove_rows_with_missing_ratings(df_copy):
    df_1 = df_copy.dropna(subset=['Cleanliness_rating', 'Accuracy_rating','Communication_rating', 'Location_rating','Check-in_rating', 'Value_rating'])
    return df_1

def combine_description_strings(df_1):
    df_2 = df_1.dropna(subset=['Description'])
    df_2["Description"] = df_2["Description"].str.replace('About this space', '')
    df_2["Description"] = df_2["Description"].str.replace(' ', '')
    df_2["Description"] = df_2["Description"].str.replace('\'', '')
    df_2["Description"] = df_2["Description"].str.replace('"', '')
    df_2["Description"] = df_2["Description"].str.replace('[', '')
    df_2["Description"] = df_2["Description"].str.replace(']', '')
    df_2["Description"] = df_2["Description"].str.join("_")
    return df_2

def set_default_feature_values(df_2):
    df_2[["guests", "beds", "bathrooms","bedrooms"]] = df_2[["guests", "beds", "bathrooms","bedrooms"]].fillna(1)
    df_3 = df_2
    return df_3

def clean_tabular_data():
    df_1 = remove_rows_with_missing_ratings(df_copy)
    df_2 = combine_description_strings(df_1)
    df_3 = set_default_feature_values(df_2)
    return df_3

if __name__ == '__main__':
    df = pd.read_csv("airbnb-property-listings/tabular_data/listing.csv")
    df_copy = df.copy()
    df_copy = df_copy.replace(r'\r+|\\n+|\t+','', regex=True)
    new_df = clean_tabular_data()
    new_df.to_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index = False)

