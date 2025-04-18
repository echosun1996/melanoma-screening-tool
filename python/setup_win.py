# conda activate ScreenToolPC
from cx_Freeze import setup, Executable
import sys
import importlib.metadata
import re
import timm
import os
timm_path = os.path.dirname(timm.__file__)

# 清理版本字符串中的无效格式
def clean_version_spec(version_spec):
    # 匹配并修复形如 >=1.8.* 的版本规范
    return re.sub(r'(>=|>)([0-9.]+)\.\*', r'>=\2.0', version_spec)

# 修补 packaging.requirements.Requirement 以处理无效格式
try:
    import packaging.requirements
    original_init = packaging.requirements.Requirement.__init__

    def patched_init(self, requirement_string):
        try:
            original_init(self, requirement_string)
        except packaging.requirements.InvalidRequirement:
            # 尝试清理并重新初始化
            cleaned_req = clean_version_spec(requirement_string)
            original_init(self, cleaned_req)

    packaging.requirements.Requirement.__init__ = patched_init
except ImportError:
    pass

# 创建可执行文件的配置
# Windows下需要设置icon和添加base选项
executableApp = Executable(
    script="main.py",
    target_name="pyapp.exe",  # 添加.exe后缀
    base="Win32GUI",  # 使用Windows GUI模式，不显示控制台窗口
    # icon="path/to/icon.ico",  # 如果有图标，可以在这里指定
)

# 标准库中需要明确包含的模块
std_lib_includes = [
    "html", "http", "email", "json", "logging", "uuid", "encodings",
    "codecs", "io", "abc", "_weakrefset", "copyreg", "_collections_abc",
    "functools", "re", "sre_constants", "sre_parse", "sre_compile", "types",
    "os", "sys", "ntpath", "stat", "genericpath", "errno"  # Windows使用ntpath而不是posixpath
]

# 打包的参数配置
options = {
    "build_exe": {
        "build_exe": "./dist/",
        "packages": [
            "flask",
            "numpy",
            "PIL",
            "cv2",
            "werkzeug",
            "jinja2",
            "itsdangerous",
            "click",
            "datetime",
            "os",
            "sys",
            "base64",
            "io",
            "time",
            "uuid",
            "json",
            "collections",
            "importlib",
            "torch",
            "models",
            "models.Tip_utils",
            "models.utils",
            "timm",
            "timm.layers",
            "timm.models",
        ],
        "excludes": [
            # 可以根据需要添加排除的包
        ],
        "includes": std_lib_includes + ["argparse"],
        "include_files": [
            "requirements.txt",
            ("checkpoint/best_model_epoch.ckpt", "checkpoint/best_model_epoch.ckpt"),
            ("checkpoint/best-model-v1.ckpt", "checkpoint/best-model-v1.ckpt"),
            (timm_path, "timm"),
        ],
        # Windows特定选项
        "zip_include_packages": ["*"],  # 将所有包都压缩到zip文件中
        "include_msvcr": True,  # 包含MSVC运行时
        # 提供替代的依赖解决方案
        "constants": {
            "IGNORE_MISSING_IMPORTS": True,
            "SKIP_DEPENDENCY_DETECTION": True
        }
    }
}

setup(
    name="MelanomaAnalysisApp",
    version="1.0",
    description="Melanoma Analysis Python Backend",
    author="Echo Sun",
    options=options,
    executables=[executableApp]
)