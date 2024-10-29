import os
import tempfile
import pandas as pd

def load_csv(out_dir):
    """
    Load CSV data, handling both in-progress and completed runs.
    For in-progress runs, concatenates header and data from temporary files.
    For completed runs, loads the final CSV directly.
    """
    # Check for temporary files
    header_path = os.path.join(out_dir, 'log_header.csv.tmp')
    data_path = os.path.join(out_dir, 'log_data.csv.tmp')
    final_path = os.path.join(out_dir, 'log.csv')

    # If both temporary files exist, we're loading from an in-progress run
    if os.path.exists(header_path) and os.path.exists(data_path):
        print("Loading from in-progress run (temporary files)")
        # Create temporary directory for concatenation
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

    # If final CSV exists, load it directly
    elif os.path.exists(final_path):
        print("Loading from completed run (final CSV)")
        df = pd.read_csv(final_path)

    else:
        raise FileNotFoundError(
            f"No valid CSV files found in {out_dir}. "
            f"Expected either {final_path} or both {header_path} and {data_path}"
        )

    return df
