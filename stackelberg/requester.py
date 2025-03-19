import random
import numpy as np


class Requester:
    def __init__(self, alpha1, alpha2, beta1, beta2):
        # 效用函数相关参数
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        # 基础价格
        self.qbasic = 0
        # 需求者收益
        self.ud = 0

    def calculate_ud(self, v, xi):
        # 计算需求者收益
        q = self.qbasic * v * sum(xi)
        self.ud = self.alpha1 * np.log(1 + self.beta1 * v) + self.alpha2 * np.log(1 + self.beta2 * sum(xi)) - q

    def optimize_qbasic(self, v, xi):
        # 优化基础价格（这里使用简单的随机搜索示例，实际可能需要更复杂的优化算法）
        best_qbasic = None
        best_ud = float('-inf')
        for _ in range(100):
            qbasic = random.uniform(0, 10)
            self.qbasic = qbasic
            self.calculate_ud(v, xi)
            if self.ud > best_ud:
                best_ud = self.ud
                best_qbasic = qbasic
        self.qbasic = best_qbasic