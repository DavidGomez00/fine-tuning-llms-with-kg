import pandas as pd
import re

def clean_df_for_plotting(df:pd.DataFrame, column_names:list, split_camel_case:bool=True):
    ''' Removes the URI and optionally spaces out CamelCase text.
    Returns a copy of the df with formatted collumns.
    '''
    if len(column_names) == 0:
        return df.copy()
    
    plotting_df = df[column_names].copy()
    for column in column_names:
        # Remove prefix
        plotting_df[column] = plotting_df[column].str.split(':').str[-1]
        # Apply CamelCase regex
        if split_camel_case:
            plotting_df[column] = plotting_df[column].str.replace(
                r'(?<!^)(?=[A-Z])', 
                ' ', 
                regex=True
            ).str.capitalize()

    return plotting_df