import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
import sys
import json
import re
import argparse
import logging
import rasterio
from textwrap import dedent
from pathlib import Path
from typing import Tuple, Union
from gguf_tensor_to_image import GGUFModel, TorchModel, Quantized, Quantized_Q8_0, Model

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def scale_tensor(tensor) -> npt.NDArray[np.float32]:
    # Formula from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    r_min, r_max = tensor.min(), tensor.max()
    t_min = 0
    t_max = 1000
    scaled_tensor = ((tensor - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min
    scale_factor = (t_max - t_min) + t_min
    return scaled_tensor, scale_factor

def have_same_extension(file1, file2):
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

def extract_tensor_from_model(model_file: Union[str, Path], tensor_name: str) -> Tuple[bool, Union[None, str], Union[None, npt.NDArray[np.float32]]]:
    """
    Extracts a tensor from a given model file based on the tensor name.

    :param model_file: Path to the model file. Can be a GGUF or PyTorch model.
    :param tensor_name: Name of the tensor to extract.
    :param model_type: Type of the model ('gguf' or 'torch'). If not provided, inferred from file extension.
    :return: A tuple containing a boolean indicating success, an error message if unsuccessful, and the extracted tensor as a NumPy array if successful.
     """

    # Initialize the model
    if model_file == "gguf" or model_file.lower().endswith(".gguf"):
        model = GGUFModel(model_file)
    elif model_file1 == "torch" or model_file1.lower().endswith(".pth"):
        model = TorchModel(model_file1)
    #elif model_file1 == "stable_diffusion"
        #model1 = StableDiffusionModel(model_file1)
    else:
        raise ValueError("Unknown Model Type")

    # Validate and retrieve the tensor
    is_valid, error_message = model.valid(tensor_name)
    if not is_valid:
        return False, error_message, None

    tensor = model.get_as_f32(tensor_name)
    return True, None, tensor

def norm_array_check(diff_array, tensor1, tensor2) -> npt.NDArray[np.float32]:
    # Define scale factor
    scale_factor = 100.0
	
    # Get min and max values of the tensor
    min_val, max_val = diff_array.min(), diff_array.max()
    logging.info(f"* Min/Max Values in diff_array: max = {max_val}, min = {min_val}")

    if min_val == max_val:
        logging.warning("* Uniform array detected, increasing precision from float32 to float64.")
		
        # Calculate numerically stable means and standard deviations.
        mean1, std_dev1 = np.mean(tensor1, dtype=np.float64), np.std(tensor1, dtype=np.float64)
        mean2, std_dev2 = np.mean(tensor2, dtype=np.float64), np.std(tensor2, dtype=np.float64)
		
        # Normalize arrays
        normalized1 = (tensor1 - mean1) / std_dev1
        normalized2 = (tensor2 - mean2) / std_dev2
		
        # Calculate the difference between normalized arrays
        diff_array = normalized1 - normalized2
		
        # Check that the difference array is normalized correctly.
        min_val, max_val = diff_array.min(), diff_array.max()
		
        # If array value is not normalized correctly by increasing precision, try to scale it as well.
        if min_val == max_val:
            logging.warning(f"* Uniform array still detected, max = {max_val}, min = {min_val}")
            logging.warning(f"* Increasing precision failed to normalize arrays. Increasing precision and scaling by {scale_factor}.")
			
			# Calculate numerically stable means and standard deviations from the scaled tensors
            mean1, std_dev1 = np.mean(tensor1 * scale_factor, dtype=np.float64), np.std(tensor1 * scale_factor, dtype=np.float64)
            mean2, std_dev2 = np.mean(tensor2 * scale_factor, dtype=np.float64), np.std(tensor2 * scale_factor, dtype=np.float64)
			
            # Normalize scaled arrays
            normalized1 = (tensor1 - mean1) / std_dev1
            normalized2 = (tensor2 - mean2) / std_dev2
			
            # Calculate the difference between normalized arrays
            diff_array = normalized1 - normalized2
			
            # Check that the scaled difference array is normalized correctly.
            min_val, max_val = diff_array.min(), diff_array.max()
			
            if min_val == max_val:
                logging.warning(f"* Uniform array still detected, max = {max_val}, min = {min_val}")
                raise ValueError(f"Array could not be normalized by precision increase and scaling by {scale_factor}.")
            else:
                logging.info("* Array normalization successful by precision increase and scaling.")
                return diff_array      
        else:
            logging.info("* Array normalization successful by precision increase.")
            return diff_array
    else:
        logging.info("* Array normalization successful.")
        return diff_array

def direct_compare_tensors(tensor1, tensor2) -> npt.NDArray[np.float32]:
    # Check is tensors are the same dimensions.
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same size")
		
    # Compare element-wise differences into a single array for visualization
    diff_array = tensor1 - tensor2
	
    # Figure out whether element-wise differences are additive or subtractive, relative to the tensor in model1
    sign_of_diff = np.sign(tensor1 - tensor2)

    logging.info(f"* Direct comparison: max diff = {diff_array.max()}, min diff = {diff_array.min()}")
	
    return diff_array

def mean_compare_tensors(tensor1, tensor2) -> npt.NDArray[np.float32]:
    # Check is tensors are the same dimensions.
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same size")

    # Calculate numerically stable means and standard deviations
    mean1, std_dev1 = np.mean(tensor1, dtype=np.float64), np.std(tensor1, dtype=np.float64)
    mean2, std_dev2 = np.mean(tensor2, dtype=np.float64), np.std(tensor2, dtype=np.float64)

    # Normalize arrays
    normalized1 = (tensor1 - mean1) / std_dev1
    normalized2 = (tensor2 - mean2) / std_dev2

    # Calculate the difference between normalized arrays
    diff_array = normalized1 - normalized2
	
    # Check if the array was normalized correctly.
    diff_array = norm_array_check(diff_array, tensor1, tensor2)

    return diff_array
	
def median_compare_tensors(tensor1, tensor2) -> npt.NDArray[np.float32]:
    # Check is tensors are the same dimensions.
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same size")
	
    # Calculate numerically stable medians and MADs (Median Absolute Deviation). MAD = median(|Yi – median(Yi)|)
    median1, mad_1 = np.median(tensor1, dtype=np.float64), np.median(np.abs(tensor1 - np.median(tensor1)))
    median2, mad_2 = np.median(tensor2, dtype=np.float64), np.median(np.abs(tensor2 - np.median(tensor2)))
    logging.info(f"* Median comparison: median_diff = {median_diff}, mad_diff = {mad_diff}")

    # Log tensor values for diagnostics
    logging.info(f"* Tensor1 ({tensor_name} from {model_file1}) stats: median = {median1}, MAD = {mad_1}")
    #logging.info(f"* Tensor2 ({tensor_name} from {model_file2}) stats: max = {np.max(tensor2)}, min = {np.min(tensor2)}")

    # Normalize arrays
    normalized1 = (tensor1 - median1) / mad_1
    normalized2 = (tensor2 - median2) / mad_2

    # Calculate the difference between normalized arrays
    diff_array = normalized1 - normalized2
	
    return diff_array

def difference_to_heatmap(color_mode, diff_array) -> Image:
    # Logging initial array info
    logging.info(f"* Input diff_array shape: {diff_array.shape}, dtype: {diff_array.dtype}")

    # Check if the array is 2D and numerical
    if len(diff_array.shape) != 2 or not np.issubdtype(diff_array.dtype, np.number):
        raise ValueError("Input diff_array must be a 2D numerical array")

    logging.info(f"* Input diff_array stats: max = {diff_array.max()}, max = {diff_array.min()}")

    # Apply colormap
    if color_mode == "grayscale":
        colormap_output = plt.cm.gray(diff_array)
    elif color_mode == "false color jet":
        colormap_output = plt.cm.jet(diff_array)
    elif color_mode == "false color vidiris":
        colormap_output = plt.cm.viridis(diff_array)
    else:
        raise ValueError("Unknown color mode.")

    logging.info(f"* Colormap output shape: {colormap_output.shape}")

    # Ensure correct shape for colormap output
    if colormap_output.ndim == 3 and colormap_output.shape[2] in [3, 4]:
        logging.info(f"* Alpha channel present. Discarding.")
        heatmap_array = colormap_output[..., :3]  # Discarding alpha channel if present
    else:
        raise ValueError("Unexpected shape for color map output")

    # Convert to 8-bit format
    heatmap_array = (heatmap_array * 255).astype(np.uint8)

    # Convert to PIL Image
    if heatmap_array.ndim != 3 or heatmap_array.shape[2] != 3:
        raise ValueError("Heatmap array must be 3-dimensional with 3 channels for RGB")

    if color_mode == "grayscale":
        heatmap_image = Image.fromarray(heatmap_array, 'RGB')
    elif color_mode == "false color rainbow":
        heatmap_image = Image.fromarray(heatmap_array, 'RGB')
    elif color_mode == "false color vidiris":
        heatmap_image = Image.fromarray(heatmap_array, 'RGB')
    else:
        raise ValueError("Unknown color mode.")

    return heatmap_image

def go(args: argparse.Namespace):
    # TODO add Stable Diffusion model support
    # Set argument variables
    color_mode = args.color_mode
    model_file1 = args.model_file1
    model_file2 = args.model_file2
    comparison_type = args.comparison_type
    tensor_name = args.tensor_name
    output_path = args.output_path
    model_file1: Model
    modle_file2: Model
    # Regex pattern to check if the models have the same ending prefix e.g. gguf, pth, etc.
    if not have_same_extension(model_file1, model_file1):
        raise ValueError("Model Prefixes Do Not Match")

    # Extract specified tensors from the models
    success1, error1, tensor1 = extract_tensor_from_model(model_file1, tensor_name)
    if not success1:
        raise ValueError(f"Error extracting tensor from {model_file1}: {error1}")

    success2, error2, tensor2 = extract_tensor_from_model(model_file2, tensor_name)
    if not success2:
        raise ValueError(f"Error extracting tensor from {model_file2}: {error2}")

    # Log tensor values for diagnostics
    logging.info(f"* Tensor1 ({tensor_name} from {model_file1}) stats: max = {np.max(tensor1)}, min = {np.min(tensor1)}")
    logging.info(f"* Tensor2 ({tensor_name} from {model_file2}) stats: max = {np.max(tensor2)}, min = {np.min(tensor2)}")

    # Add a check for identical tensors
    if np.array_equal(tensor1, tensor2):
        logging.warning("* The input tensors are identical, differences will be zero.")

    # Perform comparison based on specified type
    if comparison_type == 'direct':
        comparison_result = direct_compare_tensors(tensor1, tensor2)
    elif comparison_type == 'mean':
        comparison_result = mean_compare_tensors(tensor1, tensor2)
    elif comparison_type == 'median':
        comparison_result = median_compare_tensors(tensor1, tensor2)
    else:
        raise ValueError("Unknown Comparison Type")

    # Convert comparison result to image
    image = difference_to_heatmap(color_mode, comparison_result)
    if image is not None:
        output_file_path = str(args.output_path)  # Convert Path to string
        if not output_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            raise ValueError(f"Invalid file extension for output image: {output_file_path}")
        print(f"* Saving to: {output_file_path}")
        try:
            image.save(output_file_path)
        except Exception as e:
            print(f"* Error saving the image: {e}")

def main() -> None:
    # TODO Implement color_ramp_type argument
    parser = argparse.ArgumentParser(
        description="Produces heatmaps of differences in tensor values for LLM models (GGUF and PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Information on output modes:
              devs-*:
                overall: Calculates differences in tensor values between two models with the same foundation architecture.
                         By default, output will be a grayscale raster that has the same dimensions as the tensors.
                rows   : Same as above, except the calculation is based on rows.
                cols:  : Same as above, except the calculation is based on columns.
        """,
        ),
    )
    parser.add_argument(
        "model_file1",
        type=str,
        help="Filename for the first model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "model_file2",
        type=str,
        help="Filename for the second model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "tensor_name",
        type=str,
        help="Tensor name, must be from models with the same foundation architecture for the differences to be valid.",
    )
    parser.add_argument(
        "--comparison_type",
        choices=["mean", "median", "direct"],
        default="mean",
        help="Comparison types, Default: mean",
    )
    parser.add_argument(
        "--color_mode",
        choices=["grayscale", "false color jet", "false color vidiris", "binned coolwarm"],
        default="grayscale",
        help="Color mode, Default: grayscale",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Output path for the heatmap",
    )
    if len(sys.argv) != 7:
        print("Usage: python image_diff_heatmapper_mk2.py <model_file1> <model_file2> <tensor_name> --comparison_type=<comparison_type> --color_mode=<color_mode> --output_path=<output_path>")
        sys.exit(1)
    args = parser.parse_args()
    go(args)
	
if __name__ == "__main__":
    logging.info("* Starting heatmapper program. Ensure to test with known difference cases to validate functionality.")
    main()