# plot.py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_complex_residuals(x_hat, x_test, n_del=32, n_ang=32, fig_title=None):
    cm = matplotlib.cm.cool

    x_err = np.mean(np.reshape(x_test - x_hat, (x_test.shape[0], n_del*n_ang)), axis=1)
    x_re, x_im = np.real(x_err), np.imag(x_err)
    x_re_max, x_im_max = np.max(np.abs(x_re)), np.max(np.abs(x_im))
    origin = ([0]*x_hat.shape[0], [0]*x_hat.shape[0])

    fig, ax = plt.subplots()
    q = ax.quiver(*origin, x_re, x_im, angles='xy', scale_units='xy', scale=1, cmap=cm, alpha=0.1)
    ax.set_xlim([-x_re_max,x_re_max])
    ax.set_ylim([-x_im_max,x_im_max])
    if fig_title != None:
        ax.set_title(f"{fig_title} (mean=({np.mean(x_re):.3E}, {np.mean(x_im):.3E}j)")
        plt.savefig(f"fig/residuals/{fig_title}.png")

    # mse histogram
    x_err = np.reshape(x_test - x_hat, (x_test.shape[0], n_del*n_ang))
    mse = np.real(np.mean(np.conj(x_err)*x_err, axis=1))
    fig, ax = plt.subplots()
    ax.hist(mse, bins=20)
    if fig_title != None:
        ax.set_title(f"{fig_title}")
        plt.savefig(f"fig/mse_hist/{fig_title}.png")