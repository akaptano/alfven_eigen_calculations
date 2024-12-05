
# Data description

Both matrices are in written as "row column value" for all nonzero elements.

b_matrix.dat (inertia) and a_matrix.dat (field-bending) matricies give eigenvalue problem

$$ B = \omega^2 A $$

In ideal MHD, Hermitian eigenproblem is expected; some existing codes have (small, yet significant for eigenvalues)
symmetry-breaking errors. For such cases, symmetry is enforced:

```python
A = 0.5 * (A + A.T)
B = 0.5 * (B + B.T)
```
Resistive MHD and hybrid models do not produce Hermitian matrices.

# Code Example

Code below solves for all eigenvalues. It takes ~ hour for a smaller problem.

No properies of the eigenproblem (Hermitian => real eigenvalues, sparce block tri-diagonal) are leveraged.

Eigenvectors are then reviewed to find global modes. One such mode here has eigenvalue 175691.8 for a smaller problem.


```python
import numpy as np
import scipy.linalg

#load a_matrix.dat (A) and b_matrix.dat (B) as numpy arrays
B = InertiaMatrix('.').load_matrix().toarray()
A = FieldBendingMatrix('.').load_matrix().toarray()

A = 0.5 * (A + A.T)
B = 0.5 * (B + B.T)

eigenvalues, eigenvectors = scipy.linalg.eig(A, B)

np.savetxt('eigenvalues.txt', eigenvalues.view(float).reshape(-1, 2), fmt='%.16e', header='Real Imaginary')
np.savetxt('eigenvectors_real.txt', eigenvectors.real, fmt='%.16e', header='Eigenvectors (real part)')
np.savetxt('eigenvectors_imag.txt', eigenvectors.imag, fmt='%.16e', header='Eigenvectors (imaginary part)')

```

# Data import routines:


```python
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

```
