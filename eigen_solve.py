import numpy as np
import scipy.linalg
import scipy.sparse as sp
import os
import time
from dataclasses import dataclass, field


# Data import routines:
@dataclass
class FieldBendingMatrix:
    '''
    Field bending (sparce) matrix A from Az = lambda Bz generalized eigenvalue problem.
    Stored in a_matrix.dat output of AE3D code,
    where lambda is the square of frequency, and eigenvector z is the shear Alfven mode.
    '''
    sim_dir: str
    file_path: str = field(init=False)
    matrix_description: str = field(default_factory=lambda: "Field bending (sparce) matrix A from Az = lambda Bz generalized eigenvalue problem")
    matrix: sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, 'a_matrix.dat')
        self.matrix = self.load_matrix()

    def load_matrix(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        with open(self.file_path, 'r') as file:
            data = np.loadtxt(file, dtype=[('i', int), ('j', int), ('value', float)])

        # Adjust indices for 0-based indexing in Python (if original indices are 1-based)
        rows = data['i'] - 1
        cols = data['j'] - 1
        values = data['value']

        # Find the maximum index for matrix dimension
        size = max(np.max(rows), np.max(cols)) + 1

        # Create the COO sparse matrix
        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))

@dataclass
class InertiaMatrix:
    '''
    Inertia (sparce) matrix B from Az = lambda Bz generalized eigenvalue problem.
    Stored in a_matrix.dat output of AE3D code,
    where lambda is the square of frequency, and eigenvector z is the shear Alfven mode.
    '''
    sim_dir: str
    file_path: str = field(init=False)
    matrix_description: str = field(default_factory=lambda: "Inertia matrix B from Az = lambda Bz generalized eigenvalue problem")
    matrix: sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, 'b_matrix.dat')
        self.matrix = self.load_matrix()

    def load_matrix(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        with open(self.file_path, 'r') as file:
            data = np.loadtxt(file, dtype=[('i', int), ('j', int), ('value', float)])

        # Adjust indices for 0-based indexing in Python (original indices are 1-based)
        rows = data['i'] - 1
        cols = data['j'] - 1
        values = data['value']

        # Find the maximum index for matrix dimension
        size = max(np.max(rows), np.max(cols)) + 1

        # Create the COO sparse matrix
        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))

#load a_matrix.dat (A) and b_matrix.dat (B) as numpy arrays
t1 = time.time()
B = InertiaMatrix('Small/').load_matrix()  #.toarray()
A = FieldBendingMatrix('Small/').load_matrix()  #.toarray()

print(A.size, A.shape, A.count_nonzero(), "{0:.1f}% of entries are nonzero".format(A.count_nonzero() / A.shape[0] ** 2 * 100))
print(B.size, B.shape, B.count_nonzero(), "{0:.1f}% of entries are nonzero".format(B.count_nonzero() / B.shape[0] ** 2 * 100))

A = 0.5 * (A + A.T)
B = 0.5 * (B + B.T) * 1e8  # rescale
print(A.shape, B.shape)
t2 = time.time()
print('Total time to load the matrices = ', t2 - t1, ' s')
t1 = time.time()
# print(np.linalg.cond(A.toarray()), np.linalg.cond(B.toarray()))
# Appears A and B are not positive definite
# cholesky = np.linalg.cholesky(-B.toarray())
# print(cholesky)
s1 = sp.linalg.eigsh(A, k=A.shape[0]-1, return_eigenvectors=False, tol=1e-3)
s2 = sp.linalg.eigsh(B, k=B.shape[0]-1, return_eigenvectors=False, tol=1e-3)
from matplotlib import pyplot as plt
plt.figure()
plt.plot(s1)
plt.plot(s2)
plt.grid()
t2 = time.time()
print('Total time to compute the matrix eigenvalues = ', t2 - t1, ' s')
plt.show()

print('Beginning eigensolve: ')
t1 = time.time()
eigenvalues = sp.linalg.eigs(A, A.shape[0]-2, B, return_eigenvectors=False, tol=1e-1)
# print(eigenvalues)
plt.figure()
plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'ro')
plt.grid()
# plt.show()
t2 = time.time()
print('Total time for eigenvalue solve = ', t2 - t1, ' s')

# np.savetxt('eigenvalues.txt', eigenvalues.view(float).reshape(-1, 2), fmt='%.16e', header='Real Imaginary')
plt.show()
# np.savetxt('eigenvectors_real.txt', eigenvectors.real, fmt='%.16e', header='Eigenvectors (real part)')
# np.savetxt('eigenvectors_imag.txt', eigenvectors.imag, fmt='%.16e', header='Eigenvectors (imaginary part)')
