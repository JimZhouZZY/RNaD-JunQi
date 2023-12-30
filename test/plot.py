import matplotlib.pyplot as plt
import numpy as np

# 读取文件内容
file_path = 'loss.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 将文本中的数字转换为浮点数
numbers = [float(line.strip()) for line in lines]
xs = list(range(1,len(numbers)+1))

coefficients = np.polyfit(xs, numbers, deg=2)

# 生成拟合曲线的函数
fit_function = np.poly1d(coefficients)

# 生成拟合曲线上的新点
x_fit = np.linspace(min(xs), max(xs), 100)
y_fit = fit_function(x_fit)

# 绘制图表
plt.scatter(xs,numbers,s=1)
plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
