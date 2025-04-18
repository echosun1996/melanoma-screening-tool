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

# 确定是否为 macOS 系统
is_mac = sys.platform == "darwin"

# 创建可执行文件的配置
executableApp = Executable(
    script="main.py",
    target_name="pyapp",
)
# 标准库中需要明确包含的模块
std_lib_includes = [
    "html", "http", "email", "json", "logging", "uuid", "encodings",
    "codecs", "io", "abc", "_weakrefset", "copyreg", "_collections_abc",
    "functools", "re", "sre_constants", "sre_parse", "sre_compile", "types",
    "os", "sys", "posixpath", "stat", "genericpath", "errno"  # 添加这些模块
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
            "torch",  # Add PyTorch to packages
            "models",  # 添加 models 目录作为包进行编译
            "models.Tip_utils",  # 添加 models.Tips 子包
            "models.utils",  # 添加 models.Tips 子包
            "timm",
            "timm.layers",
            "timm.models",
        ],
        "excludes": [
            # "tkinter",
            # "unittest",
            # "pydoc_data",
            # "test",
            # Removed torch from excludes
            # "torchmetrics",
            # "lightly",
            # "lightning-bolts",
            # "pytorch-lightning",
        ],
        "includes": std_lib_includes + ["argparse"],
        # Add these to your include_files
        "include_files": [
            "requirements.txt",
            # 添加checkpoint目录下的模型文件
            ("checkpoint/best_model_epoch.ckpt", "checkpoint/best_model_epoch.ckpt"),
            ("checkpoint/best-model-v1.ckpt", "checkpoint/best-model-v1.ckpt"),
            # 添加models目录及其所有内容
            # ("models", "models"),
            (timm_path, "timm"),
        ],
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
    executables=[executableApp])