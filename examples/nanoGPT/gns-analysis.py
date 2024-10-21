import os
import pandas as pd
import tempfile
from gnstracking import unbiased_stats

def load_and_concat_csv(out_dir):
    """Load and concatenate the CSV header and data files."""
    header_path = os.path.join(out_dir, 'log_header.csv.tmp')
    data_path = os.path.join(out_dir, 'log_data.csv.tmp')

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_csv = os.path.join(temp_dir, 'combined.csv')

        # Concatenate header and data
        with open(temp_csv, 'w') as outfile:
            with open(header_path) as infile:
                outfile.write(infile.read())
            with open(data_path) as infile:
                outfile.write(infile.read())

        # Read the concatenated CSV
        df = pd.read_csv(temp_csv)

    return df

def get_aligned_columns(df, prefix='train/gns'):
    """Get aligned norm, pegn, and batch_size columns that match the prefix."""
    # Get all columns that start with the prefix
    prefix_cols = [col for col in df.columns if col.startswith(prefix)]

    # Organize columns by parameter name and type
    param_data = {}
    for col in prefix_cols:
        # Split column name into components
        parts = col.split('/')
        if len(parts) < 3:  # Ensure we have prefix/param/metric
            continue

        param_name = parts[-2]  # Get parameter name
        metric_type = parts[-1]  # Get metric type (norm, pegn, batch_size)

        if param_name not in param_data:
            param_data[param_name] = {}
        param_data[param_name][metric_type] = col

    # Verify we have complete data for each parameter
    valid_params = []
    for param, metrics in param_data.items():
        if any(s in param for s in ['wte', 'lm_head']):
            continue # skip tied weights
        if all(metric in metrics for metric in ['norm', 'pegn', 'batch_size']):
            valid_params.append(param)
        else:
            print(f"Warning: Incomplete metrics for parameter {param}, skipping")

    # Sort parameters by name for consistent ordering
    valid_params.sort()

    # Create aligned lists of column names
    norm_cols = [param_data[param]['norm'] for param in valid_params]
    pegn_cols = [param_data[param]['pegn'] for param in valid_params]
    batch_size_cols = [param_data[param]['batch_size'] for param in valid_params]

    return norm_cols, pegn_cols, batch_size_cols, valid_params

def compute_total_norms(df, prefix='train/gns'):
    """Compute total squared norms for each step."""
    # Get aligned columns
    norm_cols, pegn_cols, batch_size_cols, param_names = get_aligned_columns(df, prefix)

    if not norm_cols:
        raise ValueError(f"No valid columns found with prefix {prefix}")

    print(f"Processing {len(norm_cols)} parameters:")
    for param in param_names:
        print(f"  - {param}")

    # Verify we have a consistent batch size per step
    batch_sizes = df[batch_size_cols].nunique(axis=1)
    if not (batch_sizes == 1).all():
        raise ValueError("Inconsistent batch sizes found within steps")

    # Use the first batch size column (they should all be the same)
    batch_size = df[batch_size_cols[0]]

    # Square the norms and sum across parameters
    total_sgn = df[norm_cols].pow(2).sum(axis=1)
    total_spegn = df[pegn_cols].pow(2).sum(axis=1)

    return pd.DataFrame({
        'step': df['step'],
        'batch_size': batch_size,
        'sgn': total_sgn,
        'spegn': total_spegn
    })

def compute_ema(series, alpha=0.95):
    """Compute exponential moving average."""
    return series.ewm(alpha=alpha, adjust=False).mean()

def analyze_gns(out_dir, alpha=0.95, prefix='train/gns'):
    """Analyze GNS metrics using pandas and the unbiased estimator."""
    # Load data
    print(f"Loading data from {out_dir}")
    df = load_and_concat_csv(out_dir)

    # Compute total norms for each step
    totals = compute_total_norms(df, prefix)

    # Apply unbiased estimator
    results = pd.DataFrame({'step': totals['step']})
    results['batch_size'] = totals['batch_size']

    # Compute gtg and trsigma using the unbiased estimator
    gtg_list, trsigma_list = [], []
    for _, row in totals.iterrows():
        gtg, trsigma = unbiased_stats(row['sgn'], row['spegn'], row['batch_size'])
        gtg_list.append(gtg)
        trsigma_list.append(trsigma)

    results['gtg'] = gtg_list
    results['trsigma'] = trsigma_list

    # Compute EMAs
    results['gtg_ema'] = compute_ema(results['gtg'], alpha)
    results['trsigma_ema'] = compute_ema(results['trsigma'], alpha)

    # Compute GNS (using EMA values)
    results['gns'] = results['trsigma_ema'] / results['gtg_ema'].clip(lower=1e-6)

    # Add raw norms for reference
    results['sgn'] = totals['sgn']
    results['spegn'] = totals['spegn']

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze GNS metrics from CSV data')
    parser.add_argument('out_dir', type=str, help='Directory containing the CSV files')
    parser.add_argument('--alpha', type=float, default=0.95, help='EMA decay rate (default: 0.95)')
    parser.add_argument('--prefix', type=str, default='train/gns',
                      help='Prefix for GNS metrics in CSV (default: train/gns)')

    args = parser.parse_args()

    results = analyze_gns(args.out_dir, args.alpha, args.prefix)

    print("\nGNS Analysis Results:")
    print("====================")
    print(f"Final GNS: {results['gns'].iloc[-1]:.4f}")
    print(f"Final G^TG (EMA): {results['gtg_ema'].iloc[-1]:.4f}")
    print(f"Final tr(Σ) (EMA): {results['trsigma_ema'].iloc[-1]:.4f}")
    print(f"Average batch size: {results['batch_size'].mean():.1f}")

    # Save results
    output_path = os.path.join(args.out_dir, 'gns_analysis.csv')
    results.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
