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
if __name__ == '__main__':
    Lx, Ly = 0.15, 0.12
    grid_sizes = [20, 40, 80]
    omegas = np.arange(1.0, 1.9, 0.1)
    tol = 1e-5
    max_iter = 10000
    results = {}

    # 扫描不同网格与松弛因子
    for nx in grid_sizes:
        ny = int(nx * Ly / Lx)
        results[(nx, ny)] = {}
        print(f"Running grid {nx}×{ny}...")
        for omega in omegas:
            _, iters = solve_sor(Lx, Ly, nx, ny, omega, tol, max_iter)
            results[(nx, ny)][omega] = iters
        # 找最佳松弛因子
        best_omega = min(results[(nx, ny)], key=results[(nx, ny)].get)
        best_iters = results[(nx, ny)][best_omega]
        print(f"  Optimal omega = {best_omega:.2f}, iterations = {best_iters}\n")
        # 绘制该网格下收敛曲线
        plt.figure()
        plt.plot(list(results[(nx, ny)].keys()), list(results[(nx, ny)].values()), 'o-')
        plt.title(f'Convergence vs ω for grid {nx}×{ny}')
        plt.xlabel('ω')
        plt.ylabel('Iterations')
        plt.grid(True)
        plt.show()

    # 对于80×64网格，用其最佳松弛因子绘制等温线
    nx, ny = grid_sizes[-1], int(grid_sizes[-1] * Ly / Lx)
    opt_omega = min(results[(nx, ny)], key=results[(nx, ny)].get)
    T, _ = solve_sor(Lx, Ly, nx, ny, opt_omega, tol, max_iter)
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    X, Y = np.meshgrid(x, y)
    plt.figure()
    cs = plt.contour(X, Y, T, levels=10)
    plt.clabel(cs, inline=1, fontsize=8)
    plt.title(f'Isotherms (grid {nx}×{ny}, ω={opt_omega:.2f})')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True)
    plt.show()
