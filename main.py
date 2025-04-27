import numpy as np
import matplotlib.pyplot as plt
def solve_sor(Lx, Ly, nx, ny, omega, tol=1e-5, max_iter=10000):
  
    nx, ny = int(nx), int(ny)
    dx, dy = Lx/nx, Ly/ny
    # 初始化温度场
    T = np.ones((ny+1, nx+1)) * 20.0
    T[ny, :] = 100.0
    dx2, dy2 = dx*dx, dy*dy
    denom = 2*(dx2 + dy2)
    for it in range(1, max_iter+1):
        dmax = 0.0
        for j in range(1, ny):
            for i in range(1, nx):
                Told = T[j, i]
                Tgs = (dy2*(T[j, i+1] + T[j, i-1]) + dx2*(T[j+1, i] + T[j-1, i])) / denom
                # SOR 松弛
                T[j, i] = Told + omega * (Tgs - Told)
                dmax = max(dmax, abs(T[j, i] - Told))
        if dmax < tol:
            return T, it
    return T, max_iter
