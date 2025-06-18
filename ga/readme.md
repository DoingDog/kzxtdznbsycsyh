### 使用说明
1. **安装依赖**：
   - 运行 `pip install geneticalgorithm numpy pandas matplotlib` 安装所需库。
   
2. **运行代码**：
   - 将代码保存为 `ga_pid_tuning.py`。
   - 确保 `temperature.csv` 文件在同一目录下，包含 `time`、`temperature` 和 `volte` 列。
   - 执行 `python ga_pid_tuning.py`。

3. **输出内容**：
   - 系统辨识参数（K、T、tau）。
   - 优化后的 PID 参数（Kp、Ki、Kd）。
   - 性能指标（超调量、调节时间、稳态误差）。
   - 模型验证图和闭环响应图。

### 注意事项
- 如果优化结果不理想，可调整 GA 参数（如增大 `population_size` 或 `max_num_iteration`）。
- 确保 `temperature.csv` 文件格式正确且数据有效。
