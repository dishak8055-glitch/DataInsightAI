
import pandas as pd
import plotly.express as px


def create_histogram(df, column):

    data = df[[column]].dropna()

    if data.empty:
        return None

    fig = px.histogram(
        data,
        x=column,
        title=f"Distribution of {column}",
        nbins=20,
        template="plotly_dark"
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency"
    )

    return fig


def create_boxplot(df, column):

    data = df[[column]].dropna()

    if data.empty:
        return None

    fig = px.box(
        data,
        y=column,
        title=f"Box Plot of {column}",
        template="plotly_dark"
    )

    return fig


def create_bar_chart(df, column):

    counts = (
        df[column]
        .dropna()
        .astype(str)
        .value_counts()
        .head(15)
        .reset_index()
    )

    if counts.empty:
        return None

    counts.columns = [column, "Count"]

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Distribution of {column}",
        template="plotly_dark"
    )

    return fig


def create_correlation_heatmap(df, columns):

    if len(columns) < 2:
        return None

    correlation = df[columns].corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numerical Feature Correlation",
        template="plotly_dark"
    )

    return fig



def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Generates an interactive scatter plot with trendline."""
    cols_to_use = [x_col, y_col]
    if color_col and color_col in df.columns:
        cols_to_use.append(color_col)

    data = df[cols_to_use].dropna()
    if data.empty:
        return None

    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in data.columns else None,
        title=f"Scatter Plot: {x_col} vs {y_col}",
        template="plotly_dark"
    )
    return fig


def create_violin_plot(df, num_col, cat_col=None):
    """Generates an interactive violin/density distribution plot."""
    cols_to_use = [num_col]
    if cat_col and cat_col in df.columns:
        cols_to_use.append(cat_col)

    data = df[cols_to_use].dropna()
    if data.empty:
        return None

    fig = px.violin(
        data,
        y=num_col,
        x=cat_col if cat_col and cat_col in data.columns else None,
        box=True,
        points="all",
        title=f"Violin & Density Distribution: {num_col}",
        template="plotly_dark"
    )
    return fig
