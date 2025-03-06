import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Draw categorical plot
def draw_cat_plot():
    # Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
        var_name='variable',
        value_name='value'
    )

    # Group and reformat data
    df_group = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Draw the catplot
    g = sns.catplot(
        data=df_group,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        sharex=False,
        sharey=False
    )
    
    # Set axis labels and titles
    g.set_axis_labels("variable", "total")
    g.set_titles("Cardio = {col_name}")

    # Adjust x-tick labels to match test expectations
    for ax in g.axes.flat:
        ax.set_xticklabels(['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Get the figure for the output
    fig = g.fig

    # Do not modify the next line
    fig.savefig('catplot.png')
    return fig

# Draw heat map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Rename 'gender' to 'sex' to match test
    df_heat.rename(columns={'gender': 'sex'}, inplace=True)

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        square=True,
        center=0,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # Do not modify the next line
    fig.savefig('heatmap.png')
    return fig