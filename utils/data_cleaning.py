import pandas as pd


def remove_qualifiers_from_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Removes extra qualifiers from column

    :param df: pd.DataFrame - A pandas dataframe
    :param column: str - The name of the column to remove qualifiers form

    :return: pd.Series - A clean series with no extra qualifiers
    """
    return df[column].str.split().str.get(0)
