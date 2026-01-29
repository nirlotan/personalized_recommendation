"""
Visualization utilities for creating radar plots and other visualizations.

This module contains functions for creating radar (spider) plots to visualize
socio-demographic profiles of user groups across multiple dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional


def create_manual_radar_plot(
    data: Dict[str, list],
    category_name: str,
    get_name_from_mapping: Callable[[str], str],
    output_dir: str = 'radar_plots',
    figsize: tuple = (8, 8),
    show_plot: bool = False
) -> None:
    """
    Create a radar (spider) plot for visualizing socio-demographic profiles.
    
    This function creates a polar plot showing multiple accounts' demographic profiles
    across five metrics: political affiliation, gender, ethnicity, education, and age.
    Each account is represented by a different colored line with unique markers.
    
    Args:
        data: Dictionary mapping account names to lists of 5 metric values (0-1 range).
              Each list should contain values for: political, gender, ethnicity, 
              education, age (in that order).
              Example: {'account1': [0.8, 0.5, 0.9, 0.4, 0.8], ...}
        category_name: Name of the category being visualized (used for title and filename).
                      Example: "Musical Artists", "Politicians", etc.
        get_name_from_mapping: Function that takes an account name and returns a
                              human-readable label for display in the legend.
        output_dir: Directory where the plot image will be saved. Default: 'radar_plots'
        figsize: Tuple specifying figure size in inches (width, height). Default: (8, 8)
        show_plot: Whether to display the plot interactively. Default: False
        
    Returns:
        None. Saves the plot to '{output_dir}/{category_name}.png' and prints confirmation.
        
    Example:
        >>> data = {
        ...     'taylorswift13': [0.84, 0.51, 0.92, 0.62, 0.82],
        ...     'ALL': [0.87, 0.47, 0.83, 0.52, 0.76]
        ... }
        >>> def name_mapper(name):
        ...     return {'taylorswift13': 'Taylor Swift', 'ALL': 'Category Average'}[name]
        >>> create_manual_radar_plot(data, "Musical Artists", name_mapper)
        File Saved
    """
    # Setup Radar Chart Parameters
    metrics = {
        'political': 'Democrat',
        'gender': 'Female',
        'ethnicity': 'Caucasian',
        'education': 'Degree',
        'age': 'Over-25'
    }
    categories = [f"{k.capitalize()}\n({v})" for k, v in metrics.items()]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Define marker list for rotation and get color cycle
    markers = ['<', '^', 'X', '*', 'P', 'o', 's']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create polar subplot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot each account's profile
    for i, (account_name, values) in enumerate(data.items()):
        # Prepare data - close the loop by repeating first value
        plot_values = values + values[:1]
        
        # Get human-readable label
        label = get_name_from_mapping(account_name)
        
        # Special styling for the 'ALL' baseline (category average)
        if account_name.upper() == 'ALL':
            color = 'black'
            linestyle = 'dotted'
            marker = ''
            linewidth = 2.5
        else:
            # Assign color and marker based on index for visual distinction
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            linestyle = '-'
            linewidth = 2
        
        # Plot line and fill
        ax.plot(
            angles, plot_values,
            marker=marker,
            markersize=13,
            color=color,
            linewidth=linewidth,
            label=label,
            linestyle=linestyle
        )
        ax.fill(angles, plot_values, color=color, alpha=0)
    
    # Configure polar plot styling
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, size=12)
    ax.tick_params(axis='x', pad=20)
    
    # Set radial limits and ticks
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], color="grey", size=11)
    
    # Add title and legend
    plt.title(category_name, size=20, y=1.1, loc='left')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{output_dir}/{category_name}.png', bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print("File Saved")
