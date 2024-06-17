#
#       Análisis de fluctuaciones sin tendencias
#           sábado 15 de junio de 2024
#

import torch
import matplotlib.pyplot as plt
import matplotx
import matplotlib.animation as animation
from matplotlib import cm, colors
import matplotlib as mpl

# -----[ Plot style options ]-----
plt.rcParams['figure.dpi'] = 175
custom_colors = {"Blue": "#61AFEF", "Orange": "#D49F6E", "Green": "#98C379", "Rose": "#E06C75",
                 "Purple": "#C678DD", "Gold": "#E5C07B", "Cyan": "#36AABA", 0: "#61AFEF", 1: "#D49F6E",
                 2: "#98C379", 3: "#E06C75", 4: "#C678DD", 5: "#E5C07B", 6: "#36AABA", "LightCyan": "#56B6C2",
                 "AltOrange": "#D19A66", "Red": "#BE5046", "RoyaBlue": "#528BFF", "Gray": "#ABB2BF",
                 "LightGray": "#CCCCCC", "LightBlack": "#282C34", "Black": "#1D2025"}
plt.style.use(matplotx.styles.onedark)

# -----[ Time series ]-----

#x = torch.rand(1000)
#x = torch.randint(1, 20, (1000,)).float()
x = torch.randn(1000)
t = torch.linspace(0, 300, x.size()[0])


# -----[ DFA Method ]-----
def DFA_Method(x: torch.Tensor, t: torch.Tensor, window_size: int):
    # 1. Integrate the time series
    y = torch.cumsum(x - x.mean(), dim=0)

    # 2. Subdivide the integrated time series into windows of equal length
    window_size = window_size
    n_windows = y.size()[0] // window_size

    y_windows = y.unfold(0, window_size, window_size)
    t_epochs = t.unfold(0, window_size, window_size)

    # 3. Fit (linear regression) the integrated time series within each window
    fitted_windows = torch.zeros(n_windows, window_size)

    for epoch in range(n_windows):
        A = torch.vstack([t_epochs[epoch], torch.ones(len(t_epochs[epoch]))]).T
        m, c = torch.linalg.lstsq(A, y_windows[epoch], rcond=None)[0]
        fitted_windows[epoch] = m*t_epochs[epoch] + c

    # 4. Compute the variance of the residuals of the linear fit within each window
    deviations = y_windows - fitted_windows
    sqrd_deviations = torch.pow(deviations, 2)
    variance = torch.mean(sqrd_deviations, dim=1)
    RMS_deviation = torch.sqrt(variance)

    # 5. Average the RMS deviations over all windows
    F = torch.mean(RMS_deviation)

    return F
    # -----[ Plot ]-----
    plt.figure()
    plt.title("Time series")
    plt.plot(t, x, color=custom_colors["Blue"], label="Time series", linewidth=0.75)
    plt.plot(t, y, color=custom_colors["Orange"], label="Integrated time series", linewidth=0.75)
    plt.scatter(t, y, color=custom_colors["Orange"], s=2)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.ylim(-3, 4)
    plt.show()

    plt.figure()
    plt.title("Windows")
    plt.plot(t, y, color=custom_colors["Gray"], label="Integrated time series", zorder=1, linewidth=0.5)

    for epoch in range(n_windows):

        color = custom_colors[epoch - (epoch//7)*7 if epoch > 6 else epoch]

        plt.scatter(t_epochs[epoch], y_windows[epoch], color=color, label="Window", s=2, zorder=2)
        plt.plot(t_epochs[epoch], fitted_windows[epoch], color=color, label="Integrated time series", linewidth=0.75)
        plt.vlines(t_epochs[epoch][-1], -3, 4, color=color, linestyle=(5, (10, 3)), linewidth=0.25)

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.ylim(-3, 4)
    plt.show()


F_RMS = torch.tensor([])
Windows_sizes = torch.arange(10, 1000, 1)

for i in Windows_sizes:
    i = torch.Tensor.int(i)
    F_RMS = torch.cat([F_RMS, torch.tensor([DFA_Method(x, t, i)])])

F_RMS_log = torch.log10(F_RMS)
Windows_sizes_log = torch.log10(Windows_sizes)

A = torch.vstack([Windows_sizes_log, torch.ones(len(Windows_sizes_log))]).T
m, c = torch.linalg.lstsq(A, F_RMS_log, rcond=None)[0]

print(m)

plt.figure()
plt.title("Time series")
plt.plot(t, x, color=custom_colors["Blue"], label="Time series", linewidth=0.75)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.ylim([-10, 10])
plt.legend()
plt.show()

plt.figure()
plt.title("DFA")
plt.loglog(Windows_sizes_log, F_RMS_log, color=custom_colors["Blue"], label="DFA")
plt.grid()
plt.xlabel("Log(Window size)")
plt.ylabel("Log(F)")
#plt.xlim([1, 10**1])
#plt.yticks(fontsize=5)
#plt.ylim([-10**-1, 10**1])
plt.legend()
plt.show()
