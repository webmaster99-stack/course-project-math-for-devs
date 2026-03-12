import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


def display_distribution_plot(df: pd.DataFrame, column: str, hue: str | None = None, binwidth: int = 5, bins: int = 20) -> None:
    """
    Displays a histogram of a numerical column

    :param df: pd.DataFrame - A pandas dataframe
    :param column: str - The name od the column to be plotted
    :param hue: str - An optional parameter to break the plot down by a categorical column
    """
    if hue:
        sns.displot(
            data=df,
            x=column,
            kind="hist",
            binwidth=binwidth,
            hue=hue,
            multiple="stack",
            bins=bins
        )

    else:
        sns.displot(data=df, x=column, kind="hist", binwidth=binwidth, bins=bins)

    plt.show()


def display_count_plot(df: pd.DataFrame, column: str, hue: str | None = None) -> None:
    """
    Display a count plot of a categorical column
    :param df: pd.DataFrame - A pandas dataframe
    :param column: str - The categorical column to be plotted
    :param hue: str - An optional parameter to break the plot down by a categorical column
    """
    if hue:
        sns.countplot(data=df, y=column, hue=hue)

    else:
        sns.countplot(data=df, y=column)

    plt.show()


def column_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Displays the categories in a categorical column 
    and the number of observations per each category

    :param df: pd.DataFrame - A pandas dataframe
    :param column: str - The name of a categorical column

    :returns: pd.Series
    """
    return df[column].value_counts()
