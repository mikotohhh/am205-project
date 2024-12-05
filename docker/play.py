import pycutest
from scipy.optimize import minimize

# Load and define the problem
def main():
    # Print parameters for the problem ARGLALE
    pycutest.print_available_sif_params('ARGLALE')

    # Build the problem with specific parameters
    problem = pycutest.import_problem('ARGLALE', sifParams={'N': 100, 'M': 200})

    print("Problem built successfully!")
    print(problem)

    # Define the objective function, gradient, and Hessian
    def objective(x):
        f, g = problem.obj(x, gradient=True)
        return f, g

    # Solve using L-BFGS
    x0 = problem.x0  # Initial guess
    print(f"Initial guess: {x0}")

    # Use SciPy's L-BFGS-B optimizer
    result = minimize(
        fun=lambda x: problem.obj(x),  # Objective function
        x0=x0,  # Initial guess
        jac=lambda x: problem.obj(x, gradient=True)[1],  # Gradient
        method='L-BFGS-B',  # Optimization method
        options={'disp': True, 'maxiter': 100}  # Options
    )

    print("\nOptimization Result:")
    print(result)

    # Cleanup
    problem.close()

if __name__ == "__main__":
    main()
