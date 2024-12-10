import pycutest
from scipy.optimize import minimize
import numpy as np
import time

def bfgs(f, grad_f, x0, max_iter=10000, tol=1e-5):
    x = x0.copy()
    k = 0
    func_evals = 0

    n = len(x0)
    H = np.eye(n)  # Initialize Hessian approximation as the identity matrix

    grad = grad_f(x)
    func_evals += 1  # Initial function evaluation

    while np.linalg.norm(grad) > tol and k < max_iter:
        # Search direction
        direction = -H @ grad

        # Line search (simple backtracking line search)
        step_size = 1.0
        c = 1e-4
        while f(x + step_size * direction) > f(x) + c * step_size * np.dot(grad, direction):
            step_size *= 0.5
            func_evals += 1  # Function evaluation in line search

        func_evals += 1  # Function evaluation for step acceptance

        # Update variables
        x_new = x + step_size * direction
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

def bfgsw(f, grad_f, x0, max_iter=10000, tol=1e-5, c1=1e-4, c2=0.9):
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

def lbfgs(f, grad_f, x0, m=10, max_iter=10000, tol=1e-5):
    x = x0.copy()
    k = 0
    func_evals = 0

    n = len(x0)
    s_list = []  # Store s_{k} = x_{k+1} - x_{k}
    y_list = []  # Store y_{k} = grad_f_{k+1} - grad_f_{k}
    rho_list = []

    grad = grad_f(x)
    func_evals += 1  # Initial function evaluation

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

        # Line search (simple backtracking line search)
        step_size = 1.0
        c = 1e-4
        # count = 0
        while f(x + step_size * direction) > f(x) + c * step_size * np.dot(grad, direction):
            step_size *= 0.5
            func_evals += 1  # Function evaluation in line search
            # count += 1
            # if count == 20:
            #     break

        func_evals += 1  # Function evaluation for step acceptance

        x_new = x + step_size * direction
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

def lbfgsw(f, grad_f, x0, m=10, max_iter=10000, tol=1e-5, c1=1e-4, c2=0.9):
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


prob_list = ['DIXMAANE1','DIXMAANF','DIXMAANG','DIXMAANH','DIXMAANI1','DIXMAANJ','DIXMAANK','DIXMAANL']
for prob in prob_list:
    # print(pycutest.problem_properties(prob))
    # pycutest.print_available_sif_params(prob)

    p = pycutest.import_problem(prob, sifParams={'M':1000})
    dim = p.n
    x0 = p.x0

    # Define the objective function and its gradient
    def objective(x):
        f, _ = p.obj(x, gradient=True)
        return f

    def gradient(x):
        _, g = p.obj(x, gradient=True)
        return g

    # Run BFGS
    start_time = time.time()
    iters_bfgs, func_eval_bfgs, conv_bfgs = bfgsw(objective, gradient, x0)
    elapsed_time = time.time() - start_time
    print(f"{prob},BFGS,{dim},{iters_bfgs},{func_eval_bfgs},{elapsed_time},{conv_bfgs}")

    # Run L-BFGS
    for m in [5, 10, 20]:
        start_time = time.time()
        iters_lbfgs, func_eval_lbfgs, conv_lbfgs = lbfgsw(objective, gradient, x0, m=m)
        elapsed_time = time.time() - start_time
        print(f"{prob},L-BFGS-{m},{dim},{iters_lbfgs},{func_eval_lbfgs},{elapsed_time},{conv_lbfgs}")

    # Run Conjugate Gradient
    start_time = time.time()
    result = minimize(objective, x0, jac=gradient, method='CG', options={"gtol": 1e-5, "maxiter": 10000})
    elapsed_time = time.time() - start_time
    print(f"{prob},CG,{dim},{result.nit},{result.nfev},{elapsed_time},{result.success}")
    print()