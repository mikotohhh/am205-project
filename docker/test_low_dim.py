import numpy as np
from scipy.optimize import minimize
import time
from tqdm import tqdm

np.random.seed(42)

def bfgsw(f, grad_f, x0, max_iter=2000, tol=1e-5, c1=1e-4, c2=0.9):
    x = x0.copy()
    k = 0
    func_evals = 0

    n = len(x0)
    H = np.eye(n)  # Initialize Hessian approximation as the identity matrix

    grad = grad_f(x)
    func_evals += 1  # Initial gradient evaluation

    while np.linalg.norm(grad) > tol and k < max_iter:
        # Search direction
        direction = -H @ grad

        # Strong Wolfe line search
        alpha = 1.0
        phi0 = f(x)
        phi_prime0 = np.dot(grad, direction)
        func_evals += 1

        alpha_low, alpha_high = 0, None  # Bracketing
        count = 0
        while True and count < 20:
            phi = f(x + alpha * direction)
            func_evals += 1
            grad_new = grad_f(x + alpha * direction)
            phi_prime = np.dot(grad_new, direction)

            # Check Armijo condition
            if phi > phi0 + c1 * alpha * phi_prime0 or (alpha_high is not None and phi >= f(x + alpha_low * direction)):
                alpha_high = alpha
            # Check curvature condition
            elif abs(phi_prime) <= -c2 * phi_prime0:
                break
            # If not satisfying curvature, adjust alpha_low
            elif phi_prime > 0:
                alpha_high = alpha
            else:
                alpha_low = alpha

            # Update alpha using interpolation
            if alpha_high is not None:
                alpha = 0.5 * (alpha_low + alpha_high)
            else:
                alpha = 2 * alpha
            
            count += 1

        # Update variables
        x_new = x + alpha * direction
        grad_new = grad_f(x_new)

        # Compute s and y
        s = x_new - x
        y = grad_new - grad

        # Update H (Hessian approximation)
        if np.dot(s, y) > 1e-10:  # Ensure positive definiteness
            rho = 1.0 / np.dot(y, s)
            Hy = H @ y
            H = H + rho * np.outer(s, s) - (np.outer(Hy, s) + np.outer(s, Hy)) * rho + rho**2 * np.dot(y, Hy) * np.outer(s, s)

        # Update for the next iteration
        x = x_new
        grad = grad_new
        k += 1
    
    converged = np.linalg.norm(grad) <= tol
    return k, func_evals, converged

def lbfgsw(f, grad_f, x0, m=10, max_iter=2000, tol=1e-5, c1=1e-4, c2=0.9):
    x = x0.copy()
    k = 0
    func_evals = 0

    n = len(x0)
    s_list = []  # Store s_{k} = x_{k+1} - x_{k}
    y_list = []  # Store y_{k} = grad_f_{k+1} - grad_f_{k}
    rho_list = []

    grad = grad_f(x)
    func_evals += 1  # Initial gradient evaluation

    while np.linalg.norm(grad) > tol and k < max_iter:
        if k == 0:
            # Use negative gradient as initial direction
            direction = -grad
        else:
            # Two-loop recursion to compute the search direction
            q = grad.copy()
            alpha = []
            for i in range(len(s_list)-1, -1, -1):
                s = s_list[i]
                y = y_list[i]
                rho = rho_list[i]
                a = rho * np.dot(s, q)
                alpha.append(a)
                q = q - a * y
            r = q  # Assuming H_0 as Identity matrix
            for i in range(len(s_list)):
                s = s_list[i]
                y = y_list[i]
                rho = rho_list[i]
                beta = rho * np.dot(y, r)
                r = r + s * (alpha[len(s_list)-1 - i] - beta)
            direction = -r

        # Strong Wolfe line search
        alpha = 1.0
        phi0 = f(x)
        phi_prime0 = np.dot(grad, direction)
        func_evals += 1

        alpha_low, alpha_high = 0, None  # Bracketing
        count = 0
        while True and count < 20:
            phi = f(x + alpha * direction)
            func_evals += 1
            grad_new = grad_f(x + alpha * direction)
            phi_prime = np.dot(grad_new, direction)

            # Check Armijo condition
            if phi > phi0 + c1 * alpha * phi_prime0 or (alpha_high is not None and phi >= f(x + alpha_low * direction)):
                alpha_high = alpha
            # Check curvature condition
            elif abs(phi_prime) <= -c2 * phi_prime0:
                break
            # If not satisfying curvature, adjust alpha_low
            elif phi_prime > 0:
                alpha_high = alpha
            else:
                alpha_low = alpha

            # Update alpha using interpolation
            if alpha_high is not None:
                alpha = 0.5 * (alpha_low + alpha_high)
            else:
                alpha = 2 * alpha
            
            count += 1

        # Update variables
        x_new = x + alpha * direction
        grad_new = grad_f(x_new)

        s = x_new - x
        y = grad_new - grad

        if np.dot(s, y) > 1e-10:  # To ensure positive definiteness
            if len(s_list) == m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / np.dot(y, s))

        x = x_new
        grad = grad_new
        k += 1

    converged = np.linalg.norm(grad) <= tol
    return k, func_evals, converged

# Function to test optimization methods
def test_methods(x0, dim, method, maxcor=None, gtol=1e-5, maxiter=2000):
    if method == "L-BFGS":
        start_time = time.time()
        k, func_evals, converged = lbfgsw(func, grad, x0, m=maxcor)
        elapsed_time = time.time() - start_time
        return {
            "method": f"{method}-{maxcor}",
            "dimension": dim,
            "iterations": k,
            "func_evals": func_evals,
            "time": elapsed_time,
            "success": converged,
        }
    elif method == "BFGS":
        start_time = time.time()
        k, func_evals, converged = bfgsw(func, grad, x0)
        elapsed_time = time.time() - start_time
        return {
            "method": f"{method}",
            "dimension": dim,
            "iterations": k,
            "func_evals": func_evals,
            "time": elapsed_time,
            "success": converged,
        }
    else:
        start_time = time.time()
        result = minimize(
            func,
            x0,
            jac=grad,
            method=method,
            options={"gtol": gtol, "maxiter": maxiter},
        )
        elapsed_time = time.time() - start_time
        return {
            "method": f"{method}",
            "dimension": dim,
            "iterations": result.nit,
            "func_evals": result.nfev,
            "time": elapsed_time,
            "success": result.success,
        }

# Ackley function definition
def func(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, n + 1) * x)
    return sum1 + sum2**2 + sum2**4

# Gradient of Ackley function
def grad(x):
    n = len(x)
    indices = np.arange(1, n + 1)
    sum2 = np.sum(0.5 * indices * x)

    grad = 2 * x + (2 * sum2 + 4 * sum2**3) * 0.5 * indices
    return grad

# Test the methods
dimensions = [5]  # Example dimensions
maxcors = [5, 10, 20]
methods = ["BFGS", "L-BFGS", "CG"]  # Methods to compare
results = []

for dim in tqdm(dimensions):
    # x0 = np.array([np.random.uniform(-5,10), np.random.uniform(0,15)])
    x0 = np.random.uniform(-5, 10, dim)
    for method in methods:
        if method != "L-BFGS":
            res = test_methods(x0, dim, method)
            results.append(res)
        else:
            for maxcor in maxcors:
                res = test_methods(x0, dim, method, maxcor=maxcor)
                results.append(res)

# Display results
import pandas as pd

df_results = pd.DataFrame(results)
df_results.to_csv("zakharov.csv", index=False)
print(df_results)