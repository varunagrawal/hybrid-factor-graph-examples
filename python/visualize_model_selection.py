import gtsam
import numpy as np
import seaborn as sns
from gtsam import BetweenFactorDouble, HybridNonlinearFactor
from gtsam.symbol_shorthand import L, X
from matplotlib import pyplot as plt

sns.set_theme(style="ticks")


def get_between_factor(mus, sigmas):
    """
    Helper to get the Loop Closure factor.
    """

    factors = []
    for mu, sigma in zip(mus, sigmas):
        noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
        neg_log_constant = noise_model.negLogConstant()
        f = BetweenFactorDouble(X(0), X(10), mu, noise_model)
        factors.append((f, neg_log_constant))

    l = (L(0), len(sigmas))
    mixture_factor = HybridNonlinearFactor(l, factors)

    return mixture_factor


def plot_probability(mus, sigmas, x, errors, show=False):
    """Plot each error value in neg-exp form to show probability"""
    for i, s in enumerate(sigmas):
        plt.fill(x,
                 np.exp(-errors[i]),
                 "-",
                 label=f"$\mu_{i}={mus[i]}, \sigma_{i}={s}$",
                 alpha=0.5)
        plt.xlabel("$x = x_{t+1} - x_t$")
        plt.ylabel("p(x)")

    plt.legend(loc="upper right")
    plt.title("Probability")

    if show:
        plt.show()


def plot_loss(mus, sigmas, x, errors, show=False):
    """Plot the loss landscape for each sigma"""
    for i, s in enumerate(sigmas):
        plt.plot(x, errors[i], "-", label=f"$\mu_{i}={mus[i]}, \sigma_{i}={s}$", alpha=0.5)
        plt.xlabel("$x = x_{t+1} - x_t$")
        plt.ylabel("error")

    plt.legend(loc="upper right")
    plt.title("Loss Landscape")
    plt.grid(visible=True)

    if show:
        plt.show()


def main():
    """Main runner"""
    graph = gtsam.HybridGaussianFactorGraph()

    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([0.0001]))
    prior_factor = gtsam.PriorFactorDouble(X(0), 0.0, prior_noise_model)

    mus = np.asarray((0.0, 0.0))
    # mus = np.asarray((0.0, 0.2))
    sigmas = np.asarray((0.1, 0.05))
    # sigmas = np.asarray((0.1, 0.1))
    # sigmas = np.asarray((10, 1, 0.5))
    f = get_between_factor(mus=mus, sigmas=sigmas)

    # x_values = np.arange(-0.6, 0.8, 0.01) # for same sigmas
    x_values = np.arange(-0.6, 0.6, 0.01)

    errors = np.zeros((len(sigmas), *x_values.shape))

    # Get linearized factor, using initial values from the dataset
    initial = gtsam.Values()
    initial.insert(X(0), 0.0)
    initial.insert(X(10), 0.0)

    # graph.push_back(prior_factor.linearize(initial))
    graph.push_back(f.linearize(initial))
    graph.print()

    for i, x in enumerate(x_values):
        vv = gtsam.VectorValues()
        vv.insert(X(0), np.asarray([0.0]))
        vv.insert(X(10), np.asarray([x]))
        dv = gtsam.DiscreteValues()
        for d in range(len(sigmas)):
            dv[L(0)] = d
            hv = gtsam.HybridValues(vv, dv)
            # print(d, f.error(hv))
            errors[d][i] = graph.error(hv)
        # print(errors[0][i], errors[1][i])

    plot_loss(mus, sigmas, x_values, errors, True)
    plot_probability(mus, sigmas, x_values, errors, True)


if __name__ == "__main__":
    main()
