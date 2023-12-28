@echo off
REM gguf_compare_models_sum_stats_mk2.py

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
REM pip install -r requirements.txt

cls

set /p help="Welcome to gguf_compare_models_sum_stats_mk2.py! Look at the program's arguments before proceeding? y/n: "

REM Check if the help argument is provided
if "%help%"=="y" (
    REM python gguf_compare_models_sum_stats_mk2.py --help
    Rem echo Hope this helps!
	echo Sorry, the help page is still under construction!
    goto end
)

REM Prompt user for model file paths and Excel file path
set /p MODEL1="Enter the name of the first model file. Default path is the directory where the batch file resides: "
set /p MODEL2="Enter the name of the second model file, if applicable. Default path is the directory where the batch file resides: "
set /p FORMAT="Enter the format of the output file, json or excel. Default is excel: "
set /p OUTPUT_FILE="Enter the name of the output file, including the file extension. Default path is the directory where the batch file resides: "
set /p MAKE_STATS_ON_DIFFERENCES="Get statistics of the difference between the two models? Default is n. y/n: "

if "%MAKE_STATS_ON_DIFFERENCES%"=="y" (
    set STATS_FLAG=--make_stats_on_differences
) else (
    set STATS_FLAG=
)

python gguf_compare_models_sum_stats_mk2.py %MODEL1% %OUTPUT_FILE% %FORMAT% %STATS_FLAG% %MODEL2%

:end
echo Process completed. Press any key to exit.
pause > nul
