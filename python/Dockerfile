# 使用Python 3.9.15官方镜像作为基础
FROM python:3.9-windowsservercore

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . .

# 安装依赖项
RUN pip install --upgrade pip && \
    pip install pyinstaller && \
    pip install -r requirements.txt

# 使用PyInstaller创建可执行文件
RUN pyinstaller pyapp_win.spec

# 输出目录将是./dist/pyapp.exe