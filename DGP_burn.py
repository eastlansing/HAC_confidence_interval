from typing import Any
import numpy as np

class DGPBase:
    burn_in = 10  # Class variable
    
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def generate(self) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclass")

class AR1_iid(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        y = np.zeros((total_size, 1))
        y[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            y[t] = self.phi * y[t-1] + np.random.randn()
        return y[self.__class__.burn_in:]
    
class AR1_MDS(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            e[t] = u[t] * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class AR1_GARCH(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        u = np.random.normal(0, 1, size=(total_size, 1))
        h = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        h[0] = np.abs(np.random.normal(0, 1))
        y[0] = np.random.normal(0, 1)
        e[0] = h[0] * u[0]
        for t in range(1, total_size):
            h[t] = np.sqrt(0.1 + 0.09 * e[t-1]**2 + 0.9 * h[t-1]**2)
            e[t] = h[t] * u[t]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]


class AR1_WN(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] + u[t-1] * u[t-2]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]
    
class AR1_WN_gam_v(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        exp1 = np.random.gamma(shape=self.shape, scale=self.scale, size=(total_size, 1))
        u = exp1 - self.shape * self.scale
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] + u[t-1] * u[t-2]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class AR1_WN_gam_v_minus(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        exp1 = np.random.gamma(shape=self.shape, scale=self.scale, size=(total_size, 1))
        u = exp1 - self.shape * self.scale
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] - u[t-1] * u[t-2]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]
    
class AR1_non_md1(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            e[t] = (u[t]**2) * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class AR1_NLMA(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t-2] * u[t-1] * (u[t-2] + u[t] + 1)
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]    

class AR1_bilinear(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] + 0.5 * u[t-1] * e[t-2]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]   
        
class ARMA11_iid(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class ARMA11_MDS(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        w = np.random.normal(0, 1, size=(total_size, 1))
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        u[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            u[t] = w[t] * w[t-1]
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class ARMA11_GARCH(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        w = np.random.normal(0, 1, size=(total_size, 1))
        h = np.zeros((total_size, 1))
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        h[0] = np.abs(np.random.normal(0, 1))
        u[0] = h[0] * w[0]
        y[0] = np.random.normal(0, 1)
        e[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            h[t] = np.sqrt(0.1 + 0.09 * u[t-1]**2 + 0.9 * h[t-1]**2)
            u[t] = h[t] * w[t]
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class ARMA11_WN(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        w = np.random.normal(0, 1, size=(total_size, 1))
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        
        u[0] = np.random.normal(0, 1)
        u[1] = np.random.normal(0, 1)
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            u[t] = w[t] + w[t-1] * w[t-2]
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class ARMA11_WN_gam_v(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        exp1 = np.random.gamma(shape=self.shape, scale=self.scale, size=(total_size, 1))
        w = exp1 - self.shape * self.scale
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        
        u[0] = np.random.normal(0, 1)
        u[1] = np.random.normal(0, 1)
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            u[t] = w[t] + w[t-1] * w[t-2]
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class ARMA11_WN_gam_v_minus(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        exp1 = np.random.gamma(shape=self.shape, scale=self.scale, size=(total_size, 1))
        w = exp1 - self.shape * self.scale
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        
        u[0] = np.random.normal(0, 1)
        u[1] = np.random.normal(0, 1)
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            u[t] = w[t] - w[t-1] * w[t-2]
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]
    
class ARMA11_non_md1(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        w = np.random.normal(0, 1, size=(total_size, 1))
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        
        u[0] = np.random.normal(0, 1)
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            u[t] = (w[t]**2) * w[t-1]
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]

class ARMA11_NLMA(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        w = np.random.normal(0, 1, size=(total_size, 1))
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        
        u[0] = np.random.normal(0, 1)
        u[0] = np.random.normal(0, 1)        
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            u[t] = w[t-2] * w[t-1] * (w[t-2] + w[t] + 1)
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:]    

class ARMA11_bilinear(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.ma1)
        w = np.random.normal(0, 1, size=(total_size, 1))
        u = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        
        u[0] = np.random.normal(0, 1)
        u[0] = np.random.normal(0, 1)
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            u[t] = w[t] + 0.5 * w[t-1] * u[t-2]
            e[t] = u[t] + theta * u[t-1]
            y[t] = self.phi * y[t-1] + e[t]
        return y[self.__class__.burn_in:] 
    
class MA1_iid(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        y = np.zeros((total_size, 1))
        y[0] = np.random.normal(0, 1)
        u = np.random.normal(0, 1, size=(total_size, 1))
        for t in range(1, total_size):
            y[t] = u[t] + theta * u[t-1]
        return y[self.__class__.burn_in:]

class MA1_MDS(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        u = np.random.normal(0, 1, size=(total_size, 1))
        for t in range(1, total_size):
            e[t] = u[t] * u[t-1]
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]

class MA1_GARCH(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        u = np.random.normal(0, 1, size=(total_size, 1))
        h = np.zeros((total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        h[0] = np.abs(np.random.normal(0, 1))
        y[0] = np.random.normal(0, 1)
        e[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            h[t] = np.sqrt(0.1 + 0.09 * e[t-1]**2 + 0.9 * h[t-1]**2)
            e[t] = h[t] * u[t]
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]

class MA1_WN(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        u = np.random.normal(0, 1, size=(total_size, 1))
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] + u[t-1] * u[t-2]
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]

class MA1_gam_v(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        exp1 = np.random.gamma(shape=self.shape, scale=self.scale, size=(total_size, 1))
        u = exp1 - self.shape * self.scale
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] + u[t-1] * u[t-2]
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]

class MA1_gam_v_minus(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        exp1 = np.random.gamma(shape=self.shape, scale=self.scale, size=(total_size, 1))
        u = exp1 - self.shape * self.scale
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] - u[t-1] * u[t-2]
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]
    
class MA1_non_md1(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        for t in range(1, total_size):
            e[t] = (u[t]**2) * u[t-1]
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]

class MA1_NLMA(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t-2] * u[t-1] * (u[t-2] + u[t] + 1)
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]    

class MA1_bilinear(DGPBase):
    def generate(self) -> np.ndarray:
        total_size = self.size + self.__class__.burn_in
        theta = np.copy(self.phi)
        u = np.random.normal(0, 1, size=(total_size, 1))
        y = np.zeros((total_size, 1))
        e = np.zeros((total_size, 1))
        e[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)
        e[1] = np.random.normal(0, 1)
        y[1] = np.random.normal(0, 1)
        for t in range(2, total_size):
            e[t] = u[t] + 0.5 * u[t-1] * e[t-2]
            y[t] = e[t] + theta * e[t-1]
        return y[self.__class__.burn_in:]   