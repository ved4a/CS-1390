import numpy as np
import matplotlib.pyplot as plt

data = np.array([-0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53,
                 0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22])

k = 2
np.random.seed(42)

means = np.random.choice(data, k)
variances = np.random.random(k)
weights = np.ones(k) / k

def gaussian_pdf(x, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean)**2) / variance)

def em_algorithm(data, k, means, variances, weights, max_iters=100, tol=1e-6):
    N = len(data)
    
    for iteration in range(max_iters):
        # E-step
        responsibilities = np.zeros((N, k))
        for i in range(k):
            responsibilities[:, i] = weights[i] * gaussian_pdf(data, means[i], variances[i])
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        
        # M-step
        weights = responsibilities.mean(axis=0)
        means = (responsibilities.T @ data) / responsibilities.sum(axis=0)
        variances = np.zeros(k)
        for i in range(k):
            variances[i] = np.dot(responsibilities[:, i], (data - means[i])**2) / responsibilities[:, i].sum()
        
        # Convergence Check
        if iteration > 0 and np.all(np.abs(means - prev_means) < tol):
            print(f"Converged at iteration {iteration}")
            break
        
        prev_means = means.copy()
    
    return means, variances, weights

means, variances, weights = em_algorithm(data, k, means, variances, weights)

print("Final Means:", means)
print("Final Variances:", variances)
print("Final Weights:", weights)

x_vals = np.linspace(min(data), max(data), 1000)
y_vals = np.zeros((k, len(x_vals)))

for i in range(k):
    y_vals[i, :] = weights[i] * gaussian_pdf(x_vals, means[i], variances[i])

y_total = y_vals.sum(axis=0)
plt.hist(data, bins=10, density=True, alpha=0.6, color='g', label='Data Histogram')
for i in range(k):
    plt.plot(x_vals, y_vals[i, :], label=f'Gaussian {i+1} (mean={means[i]:.2f}, var={variances[i]:.2f})')
plt.plot(x_vals, y_total, label='Mixture Model', color='black', linestyle='dashed')
plt.legend()
plt.show()
