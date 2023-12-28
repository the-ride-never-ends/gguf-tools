import argparse
import json
import numpy as np
import numpy.typing as npt
import pandas as pd
import openpyxl
import re
import scipy.stats
import sys
import time
#import torch
from gguf_tensor_to_image import GGUFModel, Model, TorchModel, Quantized, Quantized_Q8_0
from gguf.gguf_reader import GGUFReader, ReaderField, ReaderTensor
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, TypeVar, Tuple, Union

class ModelHandler:
    """
    Handles loading and processing of machine learning models.
    Supports GGUF and Torch models.
    """
    def __init__(self, reader, model_file: Union[str, Path], model_name: str, make_stats_on_differences: bool):
        """
        Initializes the ModelHandler with the specified model file.
        """
        self.model_file = model_file
        self.model_name = model_name  # Store the model name
        self.reader = reader
        self.make_stats_on_differences = make_stats_on_differences
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

    def load_model(self):
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

    @staticmethod
    def make_diff_array(tensor1: npt.NDArray[np.float32], tensor2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Takes the difference between two specified tensors from two different input models. 
        Tensors must be of the same type, come from the same block, and be from models with the same architecture in order for the differences to be valid.
        """
        # Check is tensors are the same dimensions.
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensor from both models must be the same shape")
		
        # Compare element-wise differences into a single array.
        try:
            print("* Creating difference array...")
            diff_array = tensor1 - tensor2
            print("Difference array 'diff_array' created.")
        except Exception as e:
            raise ValueError(f"Error taking tensor difference: {e}")

        return diff_array

    def extract_tensor_from_model(self, tensor_name: str) -> Tuple[bool, Union[None, str], Union[None, npt.NDArray[np.float32]]]:
        """
        Extract a specified tensor from an input model.
        """
        # Validate and retrieve the tensor
        is_valid, error_message = self.model.valid(tensor_name)
        if not is_valid:
            return False, error_message, None

        tensor = self.model.get_as_f32(tensor_name)

        # Ensure the tensor is a NumPy array
        if not isinstance(tensor, np.ndarray):
            return False, "Tensor is not a NumPy array", None

        return True, None, tensor

    def __call__(self, index: int):
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
        sum_stats = self.calculate_extended_statistics(tensor, self.make_stats_on_differences)
        print(f"* Performing summary statistics on {tensor_name} from [{self.model_name}] associated with index {index}...")

        return sum_stats

    @staticmethod
    def calculate_neuron_statistics(tensor: npt.NDArray[np.float32], iqr_bounds: float):
        """
        Calculate statistics for neurons outside the interquartile range.
        """
        outliers = []
        for row_index, row in enumerate(tensor):
            for col_index, value in enumerate(row):
                if value < iqr_bounds[0] or value > iqr_bounds[1]:
                    sign = "negative" if value < iqr_bounds[0] else "positive"
                    abs_dist_from_iqr = min(abs(value - iqr_bounds[0]), abs(value - iqr_bounds[1]))
                    outliers.append({
                        "row": row_index,
                        "column": col_index,
                        "value": value,
                        "sign": sign,
                        "abs_dist_from_iqr": abs_dist_from_iqr
                    })
        return outliers

    @staticmethod
    def count_leading_zeros(tensor: npt.NDArray[np.float32], count_dict: dict[str, int]) -> dict[str, int]:
        # TODO: Make this method more optimized. Converting everything to strings first and THEN counting is super inefficient.
        # TODO: Increase from 8-bit to 16-bit. This will likely add a LOT of overhead and will likely slow the function significantly.

        start_time = time.time()
        # Handle values >= 1 separately

        # Initialize categories in count_dict
        for i in range(9):  # Assuming up to 8 leading zeros based on the given data
            count_dict[f"{i}_leading_0s"] = 0

        count_dict["abs_val_more_than_1"] = np.sum(tensor >= 1)

        # Focus on values < 1
        fractional_tensor = tensor[tensor < 1]

        # Convert the tensor to a NumPy array for vectorized operations
        tensor_array = np.array(fractional_tensor)

        # Convert each number to scientific notation and extract the exponent
        exponents = [int(np.format_float_scientific(f, unique=False, precision=8).split('e-')[1]) for f in fractional_tensor]

        # Adjust exponents to get the count of leading zeros (exponent - 1)
        leading_zeros = np.array(exponents) - 1

        # Count the occurrences of each count of leading zeros and update count_dict
        unique, counts = np.unique(leading_zeros, return_counts=True)
        for zero_count, count in zip(unique, counts):
            count_dict[f"{zero_count}_leading_0s"] = count

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        print(f"{count_dict} created in {elapsed_time} seconds.")
        return count_dict

    @staticmethod
    def calculate_diff_array_statistics(tensor: npt.NDArray[np.float32]) -> tuple:
        """
        Calculate additional statistics for the difference array.
        """

        # Count the number of zero values in the tensor.
        zero_count = {'count_0': np.sum(tensor == 0)}

        # Create a tensor of non-zero elements in the tensor.
        non_zero_tensor = tensor[tensor != 0]

        # Separate the non-zero tensor into positive and negative values
        positive_values = non_zero_tensor[non_zero_tensor > 0]
        negative_values = (non_zero_tensor[non_zero_tensor < 0]) * -1 # Since positive and negative values are already split into separate dictionaries, we can just turn negatives positive again to simplify things.

        # Create dictionaries to store counts for each range for positive and negative values
        positive_count_dict = {}
        negative_count_dict = {}

        # Update both dictionaries with the counting_leading_zeros function.
        positive_count_dict.update(ModelHandler.count_leading_zeros(positive_values, positive_count_dict))
        negative_count_dict.update(ModelHandler.count_leading_zeros(negative_values, negative_count_dict))

        # Append the dictionary label onto the  
        for key in list(positive_count_dict.keys()):
            positive_count_dict[f"count_pos_{key}"] = positive_count_dict[key]

        for key in list(negative_count_dict.keys()):
            negative_count_dict[f"count_neg_{key}"] = negative_count_dict[key]

        print("Dictionaries for diff_array completed.")
        return positive_count_dict, negative_count_dict, zero_count

    @staticmethod
    def calculate_extended_statistics(tensor: npt.NDArray[np.float32], make_stats_on_differences: bool):
        """
        Calculate summary statistics for the tensor layer
        """
        # Basic statistics that don't require flattening
        basic_stats = {
            'mean': np.mean(tensor), # Global mean (?)
            'std_dev': np.std(tensor),
            'max': np.max(tensor),
            'min': np.min(tensor),
            'variance': np.var(tensor),
        }

        tensor_stats = {
            "shape": tensor.shape,
            "size": tensor.size,
            "dtype": tensor.dtype,
            "rank": len(tensor.shape),
            "stride": tensor.strides,
            "contiguity": tensor.flags.contiguous,
            "mem_layout": "c_row_major" if tensor.flags['C_CONTIGUOUS'] else "f_column_major"
        }

        # Flatten the tensor for certain statistics
        tensor_flattened = tensor.flatten() if tensor.ndim > 1 else tensor

        # Additional statistics on the flattened tensor
        flattened_stats = {
            'median': np.median(tensor_flattened),
            'range': np.ptp(tensor_flattened),
            '25th_percentile': np.percentile(tensor_flattened, 25),
            '75th_percentile': np.percentile(tensor_flattened, 75),
            'iqr': scipy.stats.iqr(tensor_flattened),
            'skewness': scipy.stats.skew(tensor_flattened),
            'kurtosis': scipy.stats.kurtosis(tensor_flattened),
            # Outlier detection based on IQR
            'outliers_positive': np.sum(tensor_flattened > (np.percentile(tensor_flattened, 75) + 1.5 * scipy.stats.iqr(tensor_flattened))),
            'outliers_negative': np.sum(tensor_flattened < (np.percentile(tensor_flattened, 25) - 1.5 * scipy.stats.iqr(tensor_flattened)))
        }
        
        if make_stats_on_differences:
            positive_count_dict, negative_count_dict, zero_count = ModelHandler.calculate_diff_array_statistics(tensor)
            # Combine basic, flattened, tensor statistics, and the extended statistics for the difference array.
            stats = {**basic_stats, **flattened_stats, **tensor_stats, **positive_count_dict, **negative_count_dict, **zero_count}
        else:
            # Combine basic, flattened, and tensor statistics
            stats = {**basic_stats, **flattened_stats, **tensor_stats}

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

def main(model_file1: Union[str, Path], output_file: Union[str, Path], format: str, make_stats_on_differences: bool, model_file2: Union[str, Path]) -> None:
    """
    Main function to get summary statistics from one or two AI models, and then output them to either an Excel or a JSON file.
    """

    # Start an overall time counter.
    start_time = time.time()
    print(f"Specified Arguments: MODEL1 [{model_file1}], MODEL2 [{model_file2}], OUTPUT [{output_file}], FORMAT [{format}], MAKE_STATS_ON_DIFFERENCES [{make_stats_on_differences}]")
    print("Note: Flattened tensors are used for the following statistics: median, range, 25th_percentile, 75th_percentile, iqr, skewness, kurtosis, outliers_positive, outliers_negative.")

    if model_file1 is None:
        raise NameError("First model not specified.")

    if model_file2 is not None:
        # Regex pattern to check if the models have the same ending prefix e.g. gguf, pth, etc.
        # TODO: Fix this, as it currently just breaks.
        #if not ModelHandler.have_same_extension(model_file1, model_file2):
          #raise ValueError("Model prefixes do not match")
        # Create the reader for Model 2.
        reader2 = GGUFReader(model_file2, "r")
        # Load the handler for Model 2.
        model2_handler = ModelHandler(reader2, model_file2, f"{model_file2}", make_stats_on_differences)

    # Create the reader for Model 1.
    reader1 = GGUFReader(model_file1, "r")
    
    # Load the handler for Model 1.
    model1_handler = ModelHandler(reader1, model_file1, f"{model_file1}", make_stats_on_differences)

    # Initialize an empty list to store row data for Excel or dictionaries for JSON
    data_entries = []

    # For the index and tensor value in the first model's tensors...
    for index, tensor in enumerate(reader1.tensors):
        # Split the tensor name from Model 1 into parts
        tensor_name = reader1.get_tensor(index).name
        parts = tensor_name.split('.')
        block_number = parts[1] if len(parts) > 1 else None
        layer_type = parts[2] if len(parts) > 2 else None

        if model_file2 is not None: 
            if make_stats_on_differences:
                # Read in the two model tensors and make the difference array.
                _, _, tensor1 = model1_handler.extract_tensor_from_model(reader1.get_tensor(index).name)
                _, _, tensor2 = model2_handler.extract_tensor_from_model(reader2.get_tensor(index).name)
                diff_array = ModelHandler.make_diff_array(tensor1, tensor2)
                # Get statistics from each model and the difference array.
                print(f"* Performing summary statistics on the differences between the two model's {tensor_name} layer associated with index {index}...")
                diff_stats = ModelHandler.calculate_extended_statistics(diff_array, make_stats_on_differences)
                stats1 = model1_handler(index)
                stats2 = model2_handler(index)
            else:
                # Get statistics from each model.
                stats1 = model1_handler(index)
                stats2 = model2_handler(index)
        else:
            stats1 = model1_handler(index)

        # Create a dictionary for all the stats we calculated, then put the stats into it.
        row = {'index': index} # A counter index, starting from 0. Useful for sorting in Excel.
        row.update({'layer': reader1.get_tensor(index).name}) # The tensor's original name. Ex: blk.31.attn_q.weight
        row.update({'block': block_number}) # Block where the tensor is from. Ex: 31
        row.update({'layer_type': layer_type}) # The type of the tensor. Ex: attn_q
        row.update({f'{model_file1}_{key}': value for key, value in stats1.items()}) # The summary statistics for model 1 by tensor layer.
        if model_file2 is not None: 
            row.update({f'{model_file2}_{key}': value for key, value in stats2.items()}) # The summary statistics for model 2 by tensor layer.
            if make_stats_on_differences:
                row.update({f'diff_{key}': value for key, value in diff_stats.items()}) # The summary statistics for the difference array, including the extended statistics.
        print("* Appending summary statistics to row...")

        if format == 'excel':
            data_entries.append(row)
        elif format == 'json':
            json_entry = {
            }
            data_entries.append(json_entry)
        print("Layer complete.")

    if format == 'excel':
        # Put all the data into a Panda data frame.
        new_data = pd.DataFrame(data_entries)
        # Export the new data to the Excel file.
        excel_handler = ExcelHandler(output_file)
        excel_handler(new_data)
    elif format == 'json':
        with open(output_file, 'w') as json_file:
            # Put all the data into a Json file.
            json.dump(data_entries, json_file, indent=4)

    end_time = time.time()
    # Elapsed time since program start in minutes.
    elapsed_time = (end_time - start_time) // 60
    print(f"Program took {elapsed_time} seconds to run.")

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
        "output",
        type=str,
        help="Filename for the output file. The output file's format must match the format type.",
    )
    parser.add_argument(
        "format", 
        choices=["excel", "json"],
		default="excel",
        help="Output format: 'excel' or 'json'. Default is excel",
    )
    parser.add_argument(
        "--make_stats_on_differences", 
        action="store_true",
        default=False, 
        help="Make stats on the differences between the two models as well. Default is False",
    )
    parser.add_argument(
        "model2",
        type=str,
        help="Filename for the second model, optional, can be GGUF or PyTorch (if PyTorch support available)",
    )
    args = parser.parse_args(None if len(sys.argv) > 1 else ["Usage: python gguf_compare_models_sum_stats.py <model1> <model2> <output> <format> <make_stats_on_differences>"])

    main(args.model1, args.output, args.format, args.make_stats_on_differences, args.model2)