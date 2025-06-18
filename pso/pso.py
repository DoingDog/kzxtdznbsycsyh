import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso

# 加载数据
data = pd.read_csv('temperature.csv')
time = data['time'].values
temperature = data['temperature'].values
voltage = data['volte'].values

# 系统辨识
# 计算初始温度（前100个点的平均值）
Y_0 = np.mean(temperature[:100])
# 计算稳态温度（后100个点的平均值）
Y_ss = np.mean(temperature[-100:])
delta_Y = Y_ss - Y_0
U = 3.5  # 阶跃输入电压
K = delta_Y / U  # 稳态增益

# 使用两点法计算时间常数和滞后时间
s1 = 0.283  # 28.3% 稳态变化
s2 = 0.632  # 63.2% 稳态变化
y1 = Y_0 + s1 * delta_Y
y2 = Y_0 + s2 * delta_Y

idx1 = np.where(temperature >= y1)[0][0]
t1 = time[idx1]
idx2 = np.where(temperature >= y2)[0][0]
t2 = time[idx2]

a1 = -np.log(1 - s1)  # ≈0.333
a2 = -np.log(1 - s2)  # ≈1.0
T = (t2 - t1) / (a2 - a1)  # 时间常数
tau = t1 - a1 * T  # 滞后时间

print(f"辨识参数: K={K:.3f} °C/V, T={T:.3f} s, tau={tau:.3f} s")

# 验证模型
def simulate_fopdt(time, K, T, tau, U, Y_0):
    y = np.zeros_like(time)
    for i, t in enumerate(time):
        y[i] = Y_0 if t < tau else Y_0 + K * U * (1 - np.exp(-(t - tau) / T))
    return y

model_y = simulate_fopdt(time, K, T, tau, U, Y_0)
plt.plot(time, temperature, 'b', label='Actual Data')
plt.plot(time, model_y, 'r--', label='Identified Model')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Model Verification')
plt.legend()
plt.show()

# PID 模拟函数
def simulate_pid(Kp, Ki, Kd, T_sim=20000, dt=1, r=35):
    N = int(T_sim / dt)
    t_sim = np.arange(0, T_sim, dt)
    y = np.zeros(N)
    u = np.zeros(N)
    e = np.zeros(N)
    integral = 0
    y[0] = Y_0
    delay_steps = int(tau / dt)
    u_buffer = np.zeros(delay_steps)

    for k in range(1, N):
        e[k-1] = r - y[k-1]
        integral += e[k-1] * dt
        derivative = (e[k-1] - e[k-2]) / dt if k > 1 else 0
        u[k-1] = np.clip(Kp * e[k-1] + Ki * integral + Kd * derivative, 0, 10)

        u_delayed = u_buffer[0]
        u_buffer = np.roll(u_buffer, -1)
        u_buffer[-1] = u[k-1]
        y[k] = y[k-1] + dt * (-y[k-1] / T + K / T * u_delayed)

    e[N-1] = r - y[N-1]
    return t_sim, y, e, dt

# 优化目标函数
def objective(params):
    Kp, Ki, Kd = params
    t_sim, y, e, dt = simulate_pid(Kp, Ki, Kd)
    itae = np.sum(t_sim * np.abs(e)) * dt
    return itae

# 包装 PSO 以记录每代最佳性能
best_itae_history = []
def wrapped_objective(params):
    itae = objective(params)
    best_itae_history.append(itae)
    return itae

# PSO 优化
lb = [0, 0, 0]  # Kp, Ki, Kd 下界
ub = [20, 0.1, 5]  # Kp, Ki, Kd 上界
optimal_params, _ = pso(wrapped_objective, lb, ub, swarmsize=20, maxiter=50, debug=True)
Kp, Ki, Kd = optimal_params
print(f"优化后的 PID 参数: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")

# 打印每代最佳 ITAE
print("\n每代最佳性能指标 (ITAE):")
for i, itae in enumerate(best_itae_history[:50], 1):  # 限制到 maxiter=50
    print(f"第 {i} 代 - ITAE: {itae:.3f}")

# 评估性能
t_sim, y_opt, e_opt, dt = simulate_pid(Kp, Ki, Kd)
overshoot = max(0, max(y_opt) - 35)  # 超调量
tolerance = 0.5
settling_idx = next((i for i in range(len(y_opt)-1, -1, -1) if abs(y_opt[i] - 35) > tolerance), 0)
settling_time = t_sim[settling_idx]  # 调节时间
ss_error = abs(y_opt[-1] - 35)  # 稳态误差

print(f"\n性能指标:")
print(f"超调量: {overshoot:.3f} °C")
print(f"调节时间: {settling_time:.3f} s")
print(f"稳态误差: {ss_error:.3f} °C")

# 绘制优化后的响应曲线
plt.plot(t_sim, y_opt, 'g', label='Optimized Response')
plt.axhline(35, color='k', linestyle='--', label='Setpoint (35°C)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Closed-Loop Response with Optimized PID')
plt.legend()
plt.show()