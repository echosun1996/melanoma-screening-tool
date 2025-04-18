# 创建名为ScreenToolPC的Python 3.9.15环境
conda create -n ScreenToolPC python=3.9.15 -y

# 激活环境
conda activate ScreenToolPC

# 安装PyTorch (GPU版本，如果没有GPU，可以改用CPU版本)
# 针对Windows平台安装兼容版本
conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# 安装基础依赖包
conda install -y numpy=1.23.5 pandas=1.5.2 scikit-learn=1.2.0 tqdm=4.64.1

# 安装Flask及相关依赖
conda install -y flask flask-cors jinja2=3.1.2

# 安装OpenCV
conda install -y opencv=4.6.0 -c conda-forge

# 安装其他依赖
pip install torchmetrics==0.11.0
pip install lightly==1.2.22
pip install lightning-bolts==0.5.0
pip install pytorch-lightning==1.6.4
pip install einops==0.7.0
pip install timm==0.9.12
pip install albumentations==1.3.1
pip install tensorboard==2.11.0

# 安装cx_Freeze用于打包
pip install cx_Freeze