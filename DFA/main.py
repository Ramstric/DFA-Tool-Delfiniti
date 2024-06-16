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
plt.rcParams['figure.dpi'] = 400
custom_colors = {"Blue": "#61AFEF", "Orange": "#D49F6E", "Green": "#98C379", "Rose": "#E06C75",
                 "Purple": "#C678DD", "Gold": "#E5C07B", "Cyan": "#36AABA", 0: "#61AFEF", 1: "#D49F6E",
                 2: "#98C379", 3: "#E06C75", 4: "#C678DD", 5: "#E5C07B", 6: "#36AABA", "LightCyan": "#56B6C2",
                 "AltOrange": "#D19A66", "Red": "#BE5046", "RoyaBlue": "#528BFF", "Gray": "#ABB2BF",
                 "LightGray": "#CCCCCC", "LightBlack": "#282C34", "Black": "#1D2025"}
plt.style.use(matplotx.styles.onedark)

# -----[ Time series ]-----

x = torch.rand(100)
t = torch.linspace(0, 1, x.size()[0])

# -----[ DFA Method ]-----


# -----[ Plot ]-----
plt.plot(t, x, color=custom_colors["Blue"], label="Time series")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.ylim(-1, 2)
plt.show()

def DFA_Method():
    pass


if __name__ == '__main__':
    DFA_Method()
