
### 运行说明
1. **环境要求**：
   - 安装 Python 3.x。
   - 安装所需库：`pip install numpy pandas matplotlib pyswarm`。
2. **运行步骤**：
   - 将代码保存为 `furnace_control.py`。
   - 确保 `temperature.csv` 文件在同一目录下，且包含 `time`、`temperature` 和 `volte` 列。
   - 运行：`python furnace_control.py`。
3. **输出内容**：
   - 系统辨识参数（K、T、tau）。
   - 每代 PSO 优化过程中的最佳性能指标（ITAE）。
   - 优化后的 PID 参数（Kp、Ki、Kd）。
   - 性能指标（超调量、调节时间、稳态误差）。
   - 模型验证和优化后的闭环响应曲线。
