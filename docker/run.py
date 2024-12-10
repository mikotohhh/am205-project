import pycutest
from scipy.optimize import minimize
import multiprocessing
import time
import json
import tracemalloc


# Define the optimization methods to test
methods = ["BFGS", "L-BFGS-B", "CG"]

# Define a list of timeout values (in seconds) to experiment with
timeout_values = [1, 1.5, 2, 2.5, 3]  # Adjust as needed

# Find problems that are unconstrained, regular, and internal=False
probs = pycutest.find_problems(constraints='unconstrained', regular=True, internal=False, userN=True)
print("Found problems:", probs)


# Function to solve a single problem
def solve_problem(problem_name, method):
    try:
        # Load the problem
        problem = pycutest.import_problem(problem_name)

        # Define the objective function and gradient
        def objective(x):
            return problem.obj(x)

        def gradient(x):
            return problem.grad(x)

        # Initial guess (set to the default initial point provided by the problem)
        x0 = problem.x0

        # Measure solving time and memory usage
        tracemalloc.start()
        start_time = time.time()
        result = minimize(fun=objective, x0=x0, jac=gradient, method=method)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        solving_time = end_time - start_time
        memory_usage_peak = peak / 1024 / 1024  # Convert from bytes to megabytes
        function_evaluations = result.nfev  # Number of function evaluations

        # Cleanup problem instance
        del problem

        return {
            "success": result.success,
            "solving_time": solving_time,
            "peak_memory_usage_mb": memory_usage_peak,
            "function_evaluations": function_evaluations,
        }
    except Exception as e:
        return {
            "success": False,
            "solving_time": None,
            "peak_memory_usage_mb": None,
            "function_evaluations": None,
            "error": str(e),
        }


# Worker function to enforce time limit
def run_with_timeout(problem_name, method, timeout):
    with multiprocessing.Pool(processes=1) as pool:
        async_result = pool.apply_async(solve_problem, (problem_name, method))
        try:
            return async_result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            pool.terminate()
            return {
                "success": False,
                "solving_time": None,
                "peak_memory_usage_mb": None,
                "function_evaluations": None,
                "error": "Time limit exceeded",
            }


# Iterate through each timeout value, method, and solve all problems
for timeout in timeout_values:
    print(f"Testing with timeout: {timeout} seconds")

    for method in methods:
        print(f"Testing method: {method}")

        # Dictionary to store solving times, memory usage, and function evaluations for this method and timeout
        results = {}

        # Counters for solved and unsolved problems
        solved_count = 0
        unsolved_count = 0

        for prob_name in probs:
            print(f"Solving problem: {prob_name} with method: {method} and timeout: {timeout}")

            result = run_with_timeout(prob_name, method, timeout)
            results[prob_name] = result

            # Update counters
            if result["success"]:
                solved_count += 1
            else:
                unsolved_count += 1

            # Print result for the current problem
            print(f"Result for {prob_name} using {method}: {result}\n")

        # Calculate the solved/unsolved ratio
        total_problems = solved_count + unsolved_count
        solved_ratio = f"{solved_count}/{total_problems}" if total_problems > 0 else "0/0"

        # Add the overall metrics to the results
        results["overall_metrics"] = {
            "solved_ratio": solved_ratio,
            "total_problems": total_problems,
            "solved_count": solved_count,
            "unsolved_count": unsolved_count,
            "timeout_condition_seconds": timeout,
        }

        # Save the results for this method and timeout to a JSON file
        output_file = f"results/optimization_results_{method}_timeout_{timeout}s.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results for method {method} with timeout {timeout} seconds saved to {output_file}")
        print(f"Solved/Unsolved Ratio for {method} with timeout {timeout}: {solved_ratio}")
