# Example: building problem
import pycutest

# Print parameters for problem ARGLALE
pycutest.print_available_sif_params('ARGLALE')

# Build this problem with N=100, M=200
problem = pycutest.import_problem('ARGLALE', sifParams={'N':100, 'M':200})
print(problem)