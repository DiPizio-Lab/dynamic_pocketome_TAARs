"""This is a quick fix to plot a size category pie chart for my poster in Aachen"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def pocket_analysis_summary(df=None, csv_path=None, title="Overall Pocket Volume Distribution",
                            output_path=None, figsize=(12, 9), dpi=300):
    """
    Create a professional pie chart of pocket volume distribution using matplotlib.

    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame containing pocket data with a 'volume_category' column
    csv_path : str, optional
        Path to CSV file containing data with 'volume_category' column
    title : str, optional
        Title for the pie chart (default: "Overall Pocket Volume Distribution")
    output_path : str, optional
        If provided, save the figure to this path (e.g., 'chart.png')
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8))
    dpi : int, optional
        DPI for saved image (default: 300)

    Returns
    -------
    fig, ax : matplotlib figure and axes objects

    """

    category_counts = df['volume_category'].value_counts()
    category_order = ['Small (<250)','Medium (250-500)','Large (500-750)', 'Very Large (>750)']

    labels = []
    values = []

    for cat in category_order:
        if cat in category_counts.index:
            labels.append(cat)
            values.append(category_counts[cat])

    # old colours #DEBCCE, #C1BBDD, #BFE1D9, #BDCCDF
    colors = {'Small (<250)': '#8FA8C9',
              'Medium (250-500)': '#8FBFA6',
              'Large (500-750)': '#9A8FBF',
              'Very Large (>750)': '#D49BA8'}

    color_list = [colors[label] for label in labels]

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    total_pockets = sum(values)

    wedges, texts = ax.pie(
        values,
        labels=None,
        # labeldistance=1.1,
        # pctdistance=0.7,
        colors=color_list,
        # autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        textprops={'color': 'black', 'fontsize': 11, 'weight': 'bold', 'family': 'sans-serif'},
        # wedgeprops={'edgecolor': 'black', 'linewidth': 2}
    )
    # Manually add percentages with smart positioning
    # Percentages < threshold% go outside with connector lines
    threshold = 10  # Adjust this to change which slices get external labels

    for i, (wedge, value) in enumerate(zip(wedges, values)):
        percentage = 100 * value / total_pockets

        # Get the angle of the wedge center
        angle = (wedge.theta2 + wedge.theta1) / 2
        angle_rad = np.radians(angle)

        # Determine if we should place text inside or outside
        if percentage < threshold:
            # Place outside with connector line
            radius = 1.2  # Distance from center
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)

            ha = 'left' if x > 0 else 'right'

            ax.text(x, y, f'{percentage:.1f}%',
                    ha=ha, va='center',
                    fontsize=10, weight='bold', color='black')

            # Draw connector line from wedge edge to text
            line_start_radius = 1.0  # Outer edge of donut
            line_end_radius = radius * 0.95

            x_start = line_start_radius * np.cos(angle_rad)
            y_start = line_start_radius * np.sin(angle_rad)
            x_end = line_end_radius * np.cos(angle_rad)
            y_end = line_end_radius * np.sin(angle_rad)

            ax.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=1.5, alpha=0.6)
        else:
            # Place inside (default position)
            radius = 0.65
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)

            ax.text(x, y, f'{percentage:.1f}%',
                    ha='center', va='center',
                    fontsize=11, weight='bold', color='black')

    # Create donut hole
    centre_circle = plt.Circle((0, 0), 0.30, fc='white', edgecolor='white', linewidth=0)
    ax.add_artist(centre_circle)


    for text in texts:
        text.set_color('black')
        text.set_fontsize(11)
        text.set_weight('bold')

    ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    fig.suptitle(title, fontsize=16, weight='bold', color='black', y=0.98)
    fig.text(0.5, 0.94,f'(n={total_pockets:,} unique pockets)', ha='center', fontsize=11,color='black',style='italic')

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # plt.show()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')



def plot_stability_bar_chart(df, color_palette, output_path='stability_volcategories_pastel.png'):
    """
    Create stability bar chart from CSV pocket_analysis_summary data.
    """

    category_order = ['Small (<250)', 'Medium (250-500)', 'Large (500-750)', 'Very Large (>750)']

    df['volume_category'] = pd.Categorical(df['volume_category'],
                                           categories=category_order,
                                           ordered=True)

    stability = df.groupby(['volume_category', 'stability'], observed=True).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 8), facecolor='white')

    x = np.arange(len(stability))
    width = 0.35  # Width of each bar
    colors_list = [color_palette.get(cat, '#808080') for cat in stability.index]

    bars_stable = ax.bar(x - width / 2, stability['Stable'], width, label='Stable',
                         color=colors_list, edgecolor='black', linewidth=1.5)

    bars_transient = ax.bar(x + width / 2, stability['Transient'], width, label='Transient',
                            color=colors_list, alpha=0.5, edgecolor='black', linewidth=1.5, hatch='///')

    ax.set_ylabel('Number of Pockets', fontsize=13, weight='bold')
    ax.set_xlabel('Volume Category', fontsize=13, weight='bold')
    ax.set_title(f'Pocket Stability by Volume Category\n(n={len(df)} unique pockets)',
                 fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(stability.index, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')

    for i, (stable, transient) in enumerate(zip(stability['Stable'], stability['Transient'])):
        ax.text(i - width / 2, stable + 20, str(int(stable)), ha='center', va='bottom',
                fontsize=10, weight='bold', color='black')

        ax.text(i + width / 2, transient + 20, str(int(transient)), ha='center', va='bottom',
                fontsize=10, weight='bold', color='black')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_facecolor('white')
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()





if __name__ == "__main__":
    colours = {'Small (<250)': '#8FA8C9',
              'Medium (250-500)': '#8FBFA6',
              'Large (500-750)': '#9A8FBF',
              'Very Large (>750)': '#D49BA8'}

    df_plotting = pd.read_csv('/mnt/cpm_crienaecker/Z/RienaeckerC/apo_holo_analysis/meta_analysis/across_genes/'
                            'pocket_analysis_summary.csv')
    # Create the pie chart
    # pocket_analysis_summary(
    #     df=df_plotting,
    #     title="Overall Pocket Volume Distribution",
    #     output_path="/mnt/cpm_crienaecker/Z/RienaeckerC/apo_holo_analysis/pocket_volume_distribution.png")

    plot_stability_bar_chart(df_plotting, colours,
                             output_path='/mnt/cpm_crienaecker/Z/RienaeckerC/apo_holo_analysis/'
                                         'stability_volcategories_pastel.png')
