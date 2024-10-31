import argparse
import pandas as pd
import plotille
import numpy as np
from pathlib import Path
from csv_tools import load_csv

def plot_metric(df, metric, x_col='tokens_processed', log_y=False, log_x=False, width=80, height=20):
    """Create a plotille figure for the specified metric."""
    fig = plotille.Figure()
    fig.width = width
    fig.height = height
    fig.color_mode = 'byte'

    df = df[[x_col, metric]].dropna()
    x = df[x_col].values.astype(float)
    y = df[metric].values

    if log_x:
        x = np.log10(x)
        xlabel = f"log10({x_col})"
    else:
        xlabel = x_col

    if log_y:
        y = np.log10(y)
        ylabel = f"log10({metric})"
    else:
        ylabel = metric

    y = np.nan_to_num(y)

    fig.set_x_limits(min_=x.min(), max_=x.max())
    fig.set_y_limits(min_=y.min(), max_=y.max())

    fig.plot(x, y, lc=200, label=ylabel)
    fig.x_label = xlabel
    fig.y_label = ylabel

    return fig.show(legend=True)

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from training logs')
    parser.add_argument('log_dir', type=str, help='Directory containing the CSV files')
    parser.add_argument('metric', type=str, help='Metric to plot')
    parser.add_argument('--x-axis', type=str, default='tokens_processed',
                       help='X-axis metric (default: tokens_processed)')
    parser.add_argument('--log-y', action='store_true', help='Use log scale for y-axis')
    parser.add_argument('--log-x', action='store_true', help='Use log scale for x-axis')
    parser.add_argument('--max-tokens', type=float, help='Truncate to this many tokens')
    parser.add_argument('--width', type=int, default=80, help='Plot width')
    parser.add_argument('--height', type=int, default=20, help='Plot height')

    args = parser.parse_args()

    # Load data
    df = load_csv(args.log_dir)

    # Validate metric exists
    if args.metric not in df.columns:
        available = '\n  '.join(sorted(df.columns))
        raise ValueError(f"Metric '{args.metric}' not found in CSV. Available metrics:\n  {available}")

    # Truncate if requested
    if args.max_tokens:
        df = df[df['tokens_processed'] <= args.max_tokens]

    # Create and display plot
    plot = plot_metric(
        df,
        args.metric,
        x_col=args.x_axis,
        log_y=args.log_y,
        log_x=args.log_x,
        width=args.width,
        height=args.height
    )

    print(f"\nPlotting {args.metric} vs {args.x_axis}")
    if args.max_tokens:
        print(f"Truncated to {args.max_tokens:g} tokens")
    print()
    print(plot)

if __name__ == '__main__':
    main()
