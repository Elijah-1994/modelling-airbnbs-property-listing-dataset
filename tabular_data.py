import pandas as pd#

def remove_rows_with_missing_ratings(df_copy):
    '''
        This function creates a new pandas data frame and drops the missing values in the ratings columns.

        Returns:
            pandas data frame
            
    '''
    
    df_1 = df_copy.dropna(subset=['Cleanliness_rating', 'Accuracy_rating','Communication_rating', 
                                  'Location_rating','Check-in_rating', 'Value_rating'])
    return df_1

def combine_description_strings(df_1):
    '''
        This function creates a new pandas data frame and  processes and combines the strings in the description column.

        Returns:
            pandas data frame
            
    '''
    df_2 = df_1.dropna(subset=['Description'])
    df_2["Description"] = df_2["Description"].str.replace('About this space', '')
    df_2["Description"] = df_2["Description"].str.replace(' ', '')
    df_2["Description"] = df_2["Description"].str.replace('\'', '')
    df_2["Description"] = df_2["Description"].str.replace('"', '')
    df_2["Description"] = df_2["Description"].str.replace('[', '')
    df_2["Description"] = df_2["Description"].str.replace(']', '')
    df_2["Description"] = df_2["Description"].str.join("_")
    df_2["Description"].to_list()
    return df_2

def set_default_feature_values(df_2):
    '''
        This function creates a new pandas data frame and fills the missing values in the guests, beds, bathrooms and bedrooms columns with 1.

        Returns:
            pandas data frame
            
    '''
    df_2[["guests", "beds", "bathrooms", "bedrooms"]] = df_2[["guests", "beds", "bathrooms", "bedrooms"]].fillna(1)
    df_3 = df_2
    return df_3

def clean_tabular_data():
    '''
        This function processes the pandas data frame and returns a new data frame.

        Returns:
            pandas data frame
            
    '''
    df_1 = remove_rows_with_missing_ratings(df_copy)
    df_2 = combine_description_strings(df_1)
    df_3 = set_default_feature_values(df_2)
    return df_3

def load_airbnb(new_df) -> tuple:
    '''
        This function creates the features and labels data to be loaded into the machine learning models.

        Returns:
            tuple: returns tuple of the features and label data.
    '''
    features = new_df.select_dtypes(include=['float64', 'Int64',])
    features = features.drop('Unnamed: 19' , axis=1)
    features = features.drop('beds', axis=1)
    labels = new_df['beds']
    return features, labels

if __name__ == '__main__':
    df = pd.read_csv("airbnb-property-listings/tabular_data/listing.csv")
    df.info()
    df_copy = df.copy()
    df_copy = df_copy.replace(r'\r+|\\n+|\t+','', regex=True)
    new_df = clean_tabular_data()
    new_df.to_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index=False)
    print(new_df['Category'].value_counts(dropna=False))
    load_airbnb(new_df)
    
