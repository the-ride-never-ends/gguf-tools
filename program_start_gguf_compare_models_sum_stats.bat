@echo off
REM gguf_compare_models_sum_stats.py

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

REM Create the output directory folder if it doesn't exist
if not exist "%BASE_DIR%\output_directory" mkdir ".\output_directory"
set OUTPUT_DIRECTORY=%BASE_DIR%output_directory

REM Install requirements
pip install -r requirements.txt

cls

set /p help="Welcome to gguf_compare_models_sum_stats.py! Look at the program's arguments before proceeding? y/n: "

REM Check if the help argument is provided
if "%help%"=="y" (
    python gguf_compare_models_sum_stats.py --help
    echo Hope this helps!
    goto end
)

REM Prompt user for model file paths and Excel file path
set /p MODEL1="Enter the name of the first model file. Default path is the directory where the batch file resides: "
set /p MODEL2="Enter the name of the second model file. Default path is the directory where the batch file resides: "
set /p EXCEL="Enter the name of the Excel file. Default path is the directory where the batch file resides: "

REM Execute the Python script with provided arguments
python gguf_compare_models_sum_stats.py %MODEL1% %MODEL2% %EXCEL%

:end
echo Process completed. Press any key to exit.
pause > nul
