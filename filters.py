import numpy as np


class LMSFilter:

    def __init__(self, order: int = 64, mu: float = 0.005):
        self.order = order
        self.mu = mu

        self.w = np.zeros(self.order, dtype=np.float64)

        self.x_buf = np.zeros(self.order, dtype=np.float64)

    def update(self, x_n: float, d_n: float) -> float:

        self.x_buf[1:] = self.x_buf[:-1]

        self.x_buf[0] = x_n

        y_n = float(np.dot(self.w, self.x_buf))

        e_n = d_n - y_n

        self.w += 2.0 * self.mu * e_n * self.x_buf

        return e_n

    def process_block(self, reference, primary):

        out = np.empty(len(primary), dtype=np.float64)

        for n in range(len(primary)):
            out[n] = self.update(reference[n], primary[n])

        return out


class NLMSFilter(LMSFilter):

    def __init__(self, order: int = 64, mu: float = 0.5, eps: float = 1e-6):

        super().__init__(order=order, mu=mu)

        self.eps = eps

    def update(self, x_n: float, d_n: float) -> float:

        self.x_buf[1:] = self.x_buf[:-1]

        self.x_buf[0] = x_n

        y_n = float(np.dot(self.w, self.x_buf))

        e_n = d_n - y_n

        power = float(np.dot(self.x_buf, self.x_buf)) + self.eps

        self.w += (self.mu / power) * e_n * self.x_buf

        return e_n


class RLSFilter:

    def __init__(self, order: int = 32, lam: float = 0.99, delta: float = 1.0):

        self.order = order

        self.lam = lam

        self.delta = delta

        self.w = np.zeros(self.order, dtype=np.float64)

        self.P = np.eye(self.order, dtype=np.float64) * self.delta

        self.x_buf = np.zeros(self.order, dtype=np.float64)

    def update(self, x_n: float, d_n: float) -> float:

        self.x_buf[1:] = self.x_buf[:-1]

        self.x_buf[0] = x_n

        Px = self.P @ self.x_buf

        denom = self.lam + float(self.x_buf @ Px)

        k = Px / denom

        e_n = d_n - float(self.w @ self.x_buf)

        self.w += k * e_n

        self.P = (self.P - np.outer(k, self.x_buf @ self.P)) / self.lam

        return float(e_n)

    def process_block(self, reference, primary):

        out = np.empty(len(primary), dtype=np.float64)

        for n in range(len(primary)):
            out[n] = self.update(reference[n], primary[n])

        return out