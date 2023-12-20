import argparse
import pandas as pd
import numpy as np
import numpy.typing as npt
import scipy.stats
import re
import sys
import openpyxl
from typing import Any, Literal, NamedTuple, TypeVar, Tuple, Union
from pathlib import Path
from gguf_tensor_to_image import GGUFModel, TorchModel, Quantized, Quantized_Q8_0, Model
from gguf.gguf_reader import GGUFReader, ReaderField, ReaderTensor

class ModelHandler:
    """
    Handles loading and processing of machine learning models.
    Supports GGUF and Torch models.
    """
    def __init__(self, reader, model_file: Union[str, Path], model_name: str):
        """
        Initializes the ModelHandler with the specified model file.
        """
        self.model_file = model_file
        self.model_name = model_name  # Store the model name
        self.reader = reader
        try:
            self.model = self.load_model()
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    @staticmethod
    def have_same_extension(file1, file2):
        """
        Checks if two files have the same extension.
        """
        # Convert arguments to strings
        file1 = str(file1)
        file2 = str(file2)
	
        # Regex pattern to extract file extension
        pattern = r'\.([^.]+)$'

        # Extracting extensions
        ext1 = re.search(pattern, file1)
        ext2 = re.search(pattern, file2)

        # Check if both extensions are found and compare them
        if ext1 and ext2:
            return ext1.group(1) == ext2.group(1)
        return False

    def load_model(self: Union[str, Path]):
        """
        Loads the model based on the file extension.
        Raises an error for unsupported model types.
        """
        try:
            # Initialize the model
            if self.model_file == "gguf" or self.model_file.lower().endswith(".gguf"):
                model = GGUFModel(self.model_file)
            elif self.model_file == "torch" or self.model_file.lower().endswith(".pth"):
                model = TorchModel(self.model_file)
            elif self.model_file == "stable_diffusion":
                raise ValueError("We do not support stable diffusion models right now. Sorry!")
            else:
                raise ValueError("Unknown Model Type")
            return model
        except FileNotFoundError:
            raise ValueError("Model file not found.")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def extract_tensor_from_model(self, tensor_name: str) -> Tuple[bool, Union[None, str], Union[None, npt.NDArray[np.float32]]]:
        # Validate and retrieve the tensor
        is_valid, error_message = self.model.valid(tensor_name)
        if not is_valid:
            return False, error_message, None

        tensor = self.model.get_as_f32(tensor_name)

        # Ensure the tensor is a NumPy array
        if not isinstance(tensor, np.ndarray):
            return False, "Tensor is not a NumPy array", None

        return True, None, tensor

    def __call__(self, index):
        # Assuming each model layer can be accessed via indexing
        # Read in the tensor name.
        tensor_name = self.reader.get_tensor(index).name
        print(f"* Reading in {tensor_name} from [{self.model_name}]...")

        # Extract the specified tensor from the model.
        is_valid, error_message, tensor = self.extract_tensor_from_model(tensor_name)
        if not is_valid:
            print(f"Error extracting tensor {tensor_name} from [{self.model_name}]: {error_message}")
            return None
        print(f"* Extracting {tensor_name} from [{self.model_name}] associated with index {index}...")

        # Run summary statistics on the extracted tensor.
        sum_stats = self.calculate_extended_statistics(tensor)
        print(f"* Performing summary statistics on {tensor_name} from [{self.model_name}] associated with {index}...")

        return sum_stats

    @staticmethod
    def calculate_extended_statistics(tensor):
        # Check and flatten the tensor if necessary
        if tensor.ndim > 1:
            tensor = tensor.flatten()
            print(f"Tensor flattened. New shape: {tensor.shape}")
        stats = {
            'mean': np.mean(tensor),
            'std_dev': np.std(tensor),
            'max': np.max(tensor),
            'min': np.min(tensor),
            # The median value of the tensor. Unlike the mean, the median is not affected by extremely large or small values, making it a useful measure of central tendency, especially in skewed distributions.
            'median': np.median(tensor),
            # The squared measure of dispersion. It gives a quick sense of the spread of values in the tensor.
            'variance': np.var(tensor),
            # The difference between the maximum and minimum values. It gives a quick sense of the spread of values in the tensor.
            'range': np.ptp(tensor),
            # Percentiles (such as the 25th, 50th, and 75th) give insights into the distribution of data. For example, the 25th and 75th percentiles give a sense of the middle 50% of your data.
            '25th_percentile': np.percentile(tensor, 25),
            '75th_percentile': np.percentile(tensor, 75),
            # The difference between the 75th and 25th percentiles. It's a measure of statistical dispersion and can be useful in identifying outliers.
            'iqr': np.percentile(tensor, 75) - np.percentile(tensor, 25),
            'skewness': scipy.stats.skew(tensor),
            'kurtosis': scipy.stats.kurtosis(tensor)
        }
        return stats

class ExcelHandler:
    """
    Handles operations related to Excel file processing.
    """
    def __init__(self, excel_file: Union[str, Path]):
        self.excel_file = excel_file

    def load_existing_data(self):
        try:
            pd.read_excel(self.excel_file)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

    def save_data(self, data):
        try:
            data.to_excel(self.excel_file, index=False)
        except Exception as e:
            raise ValueError(f"Error saving to Excel file: {e}")

    def __call__(self, new_data):
        existing_data = self.load_existing_data()
        updated_data = pd.concat([existing_data, new_data])
        self.save_data(updated_data)
        print(f"* Saving data to {self.excel_file}...")

def main(model_file1: Union[str, Path], model_file2: Union[str, Path], excel_file: Union[str, Path]):
    """
    Main function to compare statistics of two models and output to an Excel file.
    """
    # Regex pattern to check if the models have the same ending prefix e.g. gguf, pth, etc.
    if not ModelHandler.have_same_extension(model_file1, model_file2):
        raise ValueError("Model Prefixes Do Not Match")

    print(f"Specified Arguments: MODEL1 [{model_file1}], MODEL2 [{model_file2}], EXCEL [{excel_file}]")

    # Create readers for both models
    reader1 = GGUFReader(model_file1, "r")
    reader2 = GGUFReader(model_file2, "r")

    # Load the handlers for both models and the Excel file.
    model1_handler = ModelHandler(reader1, model_file1, f"{model_file1}")
    model2_handler = ModelHandler(reader2, model_file2, f"{model_file2}")
    excel_handler = ExcelHandler(excel_file)

    # Initialize an empty list to store row data
    rows = []

    # For the index and tensor value in the first model's tensors...
    for index, tensor in enumerate(reader1.tensors):
        stats1 = model1_handler(index)
        stats2 = model2_handler(index)

        # Create a dictionary for the current row and append it to the list
        row = {'Index': index}
        row.update({'Layer': reader1.get_tensor(index).name})
        row.update({f'{model_file1} {key}': value for key, value in stats1.items()})
        row.update({f'{model_file2} {key}': value for key, value in stats2.items()})
        print("* Appending summary statistics to row...")
        rows.append(row)
        print("Layer complete.")

    # Create the DataFrame from the list of rows
    new_data = pd.DataFrame(rows)

    # Export the new data to the Excel file.
    excel_handler(new_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Statistics to Excel",
    )
    parser.add_argument(
        "model1",
        type=str,
        help="Filename for the first model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "model2",
        type=str,
        help="Filename for the second model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "excel",
        type=str,
        help='Filename for the Excel file',
    )
    if len(sys.argv) != 4:
        print("Usage: python gguf_compare_models_sum_stats.py <model1> <model2> <excel>")
        sys.exit(1)
    args = parser.parse_args()

    main(args.model1, args.model2, args.excel)