import unittest
import tempfile
import os
import csv
from collections import namedtuple

# Import the LogWrapper class
from tracker import LogWrapper

def convert_empty_to_none(d):
    return {k: (None if v == '' else v) for k, v in d.items()}

class TestLogWrapper(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Mock config
        mock_config = {'learning_rate': 0.01, 'batch_size': 32}

        # Create LogWrapper instance
        self.wrapper = LogWrapper(logf=lambda x: None, config=mock_config, out_dir=self.test_dir)

    def test_csv_logging_with_varying_entries(self):
        # Log data with different entries on each iteration
        self.wrapper.log({'loss': 0.5, 'accuracy': 0.8})
        self.wrapper.step()

        self.wrapper.log({'loss': 0.4, 'accuracy': 0.85, 'learning_rate': 0.01})
        self.wrapper.step()

        self.wrapper.log({'loss': 0.3, 'val_loss': 0.35})
        self.wrapper.step()

        # Close the wrapper to ensure all data is written and CSV is finalized
        self.wrapper.close()

        # The finalized CSV file path
        csv_final_path = os.path.join(self.test_dir, 'log.csv')

        # Read and verify the CSV contents
        with open(csv_final_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [convert_empty_to_none(row) for row in reader]

            # Check the number of rows
            self.assertEqual(len(rows), 3)

            # Check the headers
            expected_headers = {'step', 'loss', 'accuracy', 'learning_rate', 'val_loss'}
            self.assertEqual(set(reader.fieldnames), expected_headers)

            # Check the contents of each row
            self.assertEqual(rows[0], {'step': '0', 'loss': '0.5', 'accuracy': '0.8', 'learning_rate': None, 'val_loss': None})
            self.assertEqual(rows[1], {'step': '1', 'loss': '0.4', 'accuracy': '0.85', 'learning_rate': '0.01', 'val_loss': None})
            self.assertEqual(rows[2], {'step': '2', 'loss': '0.3', 'accuracy': None, 'learning_rate': None, 'val_loss': '0.35'})

        # Print the actual contents of rows[0] for debugging
        print("Actual contents of rows[0]:", rows[0])

    def tearDown(self):
        # Clean up the temporary directory
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()
