# 在Google Colab中运行Flask应用的说明

## 方法1：使用ngrok（推荐）

### 步骤1：安装依赖
在Colab的第一个单元格中运行：
```python
# 安装必要的包
!pip install pyngrok flask flask-socketio torch torchvision pillow numpy

# 设置ngrok认证token（可选，但推荐）
# 你需要从 https://ngrok.com/ 注册账号获取token
# !ngrok config add-authtoken YOUR_TOKEN_HERE
```

### 步骤2：上传文件
将以下文件上传到Colab：
- `app.py`（已修改支持Colab）
- `model.py`
- `config.py`
- `utils.py`
- `templates/index.html`
- `network/` 文件夹（包含所有模型文件）
- `ckpt/` 文件夹（包含检查点文件）

### 步骤3：运行应用
```python
# 运行Flask应用
python app.py
```

应用启动后，你会看到类似这样的输出：
```
应用已启动！
本地地址: http://localhost:8080
公网地址: https://xxxxx.ngrok.io
请在浏览器中打开: https://xxxxx.ngrok.io
```

### 步骤4：访问应用
复制输出的ngrok URL（如 `https://xxxxx.ngrok.io`）到浏览器中打开即可使用。

## 方法2：使用Colab端口转发

如果不想使用ngrok，也可以使用Colab的内置端口转发功能：

### 步骤1：运行应用
```python
# 运行应用（不安装pyngrok）
python app.py
```

### 步骤2：设置端口转发
在Colab中，点击左侧的"连接"图标，然后：
1. 选择"端口转发"
2. 添加端口：8080
3. 点击"打开"

### 步骤3：访问应用
Colab会提供一个公网URL，点击即可访问。

## 注意事项

1. **模型文件大小**：确保所有必要的模型文件都已上传到Colab
2. **内存限制**：Colab有内存限制，如果模型太大可能需要升级到Colab Pro
3. **会话超时**：Colab会话会在空闲一段时间后超时，需要重新运行
4. **ngrok限制**：免费版ngrok有连接数限制，付费版更稳定

## 故障排除

### 问题1：pyngrok安装失败
```python
# 尝试使用conda安装
!conda install -c conda-forge pyngrok
```

### 问题2：端口被占用
```python
# 检查端口使用情况
!lsof -i :8080
# 或者修改app.py中的端口号
```

### 问题3：模型加载失败
确保所有依赖文件都已正确上传，特别是：
- 模型权重文件
- 配置文件
- 网络结构文件

## 性能优化建议

1. **使用GPU**：在Colab中启用GPU加速
2. **减少推理时间**：调整模型参数或使用更小的模型
3. **缓存结果**：对重复请求进行缓存
4. **异步处理**：使用异步IO提高并发性能

