
import numpy as np

def pgf_from_pmf(pmf):
    # pmf: dict {deg: prob}
    degs = np.array(sorted(pmf))
    probs = np.array([pmf[d] for d in degs], dtype=float)
    def D(x):
        x = np.asarray(x)
        return (probs * (x[..., None] ** degs)).sum(axis=-1)
    def Dp(x):
        x = np.asarray(x)
        return (probs * degs * (x[..., None] ** (degs - 1))).sum(axis=-1)
    return D, Dp

def expected_degree(pmf):
    return sum(d*p for d,p in pmf.items())

def Phi(alpha, D, K, Kp, d_mean, k_mean):
    # matches the expression in the paper (use their exact version you choose)
    # Here we implement the form stated near Theorem 1.1:
    # Φ(α) = D(1 - K'(α)/k) - (d/k) * (1 - K(α) - (1-α)K'(α))
    return D(1 - Kp(alpha)/k_mean) - (d_mean/k_mean)*(1 - K(alpha) - (1-alpha)*Kp(alpha))

def maximize_on_unit_interval(f, grid=20001):
    xs = np.linspace(0.0, 1.0, grid)
    vals = f(xs)
    i = int(np.argmax(vals))
    return xs[i], vals[i]

# Example usage:
pmf_d = {0:0.0, 3:1.0}  # d=3 deterministic (edit to your distribution)
pmf_k = {4:1.0}        # k=4 deterministic

d_mean = expected_degree(pmf_d)
k_mean = expected_degree(pmf_k)

D, Dp = pgf_from_pmf(pmf_d)
K, Kp = pgf_from_pmf(pmf_k)

alpha_star, phi_star = maximize_on_unit_interval(lambda a: Phi(a, D, K, Kp, d_mean, k_mean))
rank_rate = 1 - phi_star
null_rate = phi_star

print("alpha* =", alpha_star)
print("max Phi =", phi_star)
print("pred rank/n =", rank_rate)
print("pred null/n =", null_rate)
