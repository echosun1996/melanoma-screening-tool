# conda activate ScreenToolPC
from cx_Freeze import setup, Executable

# 创建可执行文件的配置
executableApp = Executable(
    script="main.py",
    target_name="pyapp",
)

# 打包的参数配置
options = {
    "build_exe": {
        "build_exe":"./dist/",
        "excludes": ["*.txt"],
        "optimize": 2,
        "include_files": ["requirements.txt"],  # Include requirements file for documentation
        "packages": [
            "flask",
            "numpy",
            "PIL",
            "cv2",
            "logging",
            "argparse",
            "datetime"
        ],  # Make sure required packages are included
        "includes": ["argparse"]  # Ensure command line argument parsing works
    }
}

setup(
    name="pyapp",
    version="1.0",
    description="Melanoma Analysis Python Backend",
    author="Echo Sun",
    options=options,
    executables=[executableApp]
)