@echo off
REM image_diff_heatmapper_mk2.py

REM Change the directory to where your Python script resides, if needed
cd /D "%~dp0"

REM Append the Windows system32 directory to the PATH variable to ensure access to core system utilities from the command line
set PATH=%PATH%;%SystemRoot%\system32

REM Create virtual environment if it doesn't exist
if not exist "myenv" (
    echo Creating virtual environment...
    python -m venv myenv
)

REM Activate virtual environment
call myenv\Scripts\activate.bat

REM Get the directory where the batch file resides
set BASE_DIR=%~dp0

Rem Create and set various directories
if not exist "%BASE_DIR%\output_directory" mkdir ".\output_directory"
set OUTPUT_DIRECTORY=%BASE_DIR%\output_directory

REM Install requirements
pip install -r requirements.txt

set /p help="Welcome to image_diff_heatmapper_mk2.py! Look at the program's arguments before proceeding? y/n: "

REM Check if the help argument is provided
if "%help%"=="y" (
    REM call python gguf-tensor-to-image.py --help
	REM echo Hope this helps!
	echo Sorry, but this program's help page is still under construction. Sorry about that!
)

cls
:start

REM Set the first model name
set /p MODEL1="Enter the first model name to proceed: "

REM Check if a layer name is provided
if "%MODEL1%"=="" (
    echo Please provide a model name.
	pause
	exit
)

REM 
set /p MODEL2="Enter the second model name to proceed: "

REM Check if a layer name is provided
if "%MODEL2%"=="" (
    echo Please provide a model name.
	pause
	exit
)

REM  e.g. blk.21.attn_k.weight from llama-2-7b and llama-2-Chat-7b
set /p TENSOR_NAME="Enter the tensor name. Tensor names need to come from models with the same foundation architecture in order for comparisons to be valid: "

REM Check if a layer name is provided
if "%TENSOR_NAME%"=="" (
    echo Please provide a tensor name.
	pause
	exit
)

REM Choose the comparison type.
set /p COMPARISON_TYPE="Choose your comparison type: direct, mean, or median. Note: At this time, only mean is implemented: "

REM Check if a comparison type is provided
if "%COMPARISON_TYPE%"=="" (
    set "COMPARISON_TYPE=mean"
) else (
    set "COMPARISON_TYPE=%COMPARISON_TYPE%"
)

REM Choose the color mode.
set /p COLOR_MODE="Enter your color mode: grayscale, false color jet, false color vidiris, or binned coolwarm. Default: grayscale: "

REM Check if a color mode is provided
if "%COLOR_MODE%"=="" (
    set "COLOR_MODE=grayscale"
) else (
    set "COLOR_MODE=%COLOR_MODE%"
)

REM Get the output images name and format from the user
set /p OUTPUT_PATH="Enter an image name and format to proceed. If no name is provided, the image will be a PNG image named after a concatination of the comparison type, the first and second model names, and tensor name: "

REM Set the output images to the model name and layer name by default
if "%OUTPUT_PATH%"=="" (
    set "OUTPUT_PATH=%COMPARISON_TYPE%_%MODEL1%_%MODEL2%_%TENSOR_NAME%.png"
) else (
	set "OUTPUT_PATH=%OUTPUT_PATH%"
)

Rem Call the program
call python image_diff_heatmapper_mk2.py %MODEL1% %MODEL2% %TENSOR_NAME% --comparison_type=%COMPARISON_TYPE% --color_mode=%COLOR_MODE% --output_path=%OUTPUT_DIRECTORY%\%OUTPUT_PATH%
echo Done!

REM Loop back to the start
set choice=
set /p choice="Do you want to make another image? Press 'y' and enter for Yes: "
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='y' goto start