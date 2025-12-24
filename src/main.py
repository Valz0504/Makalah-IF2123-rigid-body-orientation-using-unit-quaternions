import numpy as np
from time import perf_counter

TEST_CASES = {
    # Test Case 1: Rotasi kecil, banyak iterasi
    1: {
        'name': 'Small angle, many iterations',
        'axis': [1.0, 2.0, 3.0],
        'angle_deg': 5.0,
        'n_iterations': 50000,
        'n_rotations_cost': 1000
    },
    # Test Case 2: Rotasi besar, iterasi sedang
    2: {
        'name': 'Large angle, medium iterations',
        'axis': [0.0, 1.0, 0.0],
        'angle_deg': 45.0,
        'n_iterations': 10000,
        'n_rotations_cost': 5000
    },
    # Test Case 3: Rotasi arbitrary axis, iterasi sangat banyak
    3: {
        'name': 'Arbitrary axis, stress test',
        'axis': [1.0, 1.0, 1.0],
        'angle_deg': 1.0,
        'n_iterations': 100000,
        'n_rotations_cost': 10000
    }
}

ACTIVE_TEST_CASE = 3

class Quaternion:
    
    def __init__(self, w, x, y, z):
        """Initialize quaternion q = w + xi + yj + zk"""
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    @staticmethod
    def from_axis_angle(axis, angle):
        """Bangun unit quaternion dari axis-angle"""
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis) 
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis
        return Quaternion(w, xyz[0], xyz[1], xyz[2])
    
    def conjugate(self):
        """Return conjugate q* = w - xi - yj - zk"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self):
        """Return norm ||q||"""
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """Normalize quaternion terhadap unit length"""
        n = self.norm()
        self.w /= n
        self.x /= n
        self.y /= n
        self.z /= n
    
    def multiply(self, other):
        """Quaternion multiplication: self * other"""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)
    
    def rotate_vector(self, v):
        """Rotate vector v menggunakan rumus q*v*q^(-1)"""

        p = Quaternion(0, v[0], v[1], v[2])
        
        q_conj = self.conjugate()
        result = self.multiply(p).multiply(q_conj)
        
        return np.array([result.x, result.y, result.z])

def rotation_matrix_from_axis_angle(axis, angle):
    """Bangun 3x3 rotation matrix dari axis-angle"""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def check_orthogonality(R):
    """Mengecek deviation dari orthogonality: ||R^T R - I||"""
    deviation = np.linalg.norm(R.T @ R - np.eye(3))
    return deviation

def experiment_numerical_stability(test_case):
    """Membandingkan numerical stability antara quaternions dan rotation matrix"""
    print("=" * 70)
    print("NUMERICAL STABILITY UNDER REPEATED ROTATIONS")
    print("=" * 70)
    
    axis = np.array(test_case['axis'])
    angle = np.radians(test_case['angle_deg'])
    n_iterations = test_case['n_iterations']
    
    # Initial vector
    v0 = np.array([1.0, 0.0, 0.0])
    
    # Bangun quaternion dan rotation matrix
    q = Quaternion.from_axis_angle(axis, angle)
    R = rotation_matrix_from_axis_angle(axis, angle)
    
    # compute bagian quaternion
    v_quat = v0.copy()
    for i in range(n_iterations):
        v_quat = q.rotate_vector(v_quat)
        if i % 500 == 0: 
            q.normalize()
    
    # compute bagian rotation matrix
    v_mat = v0.copy()
    R_accumulated = R.copy()
    for i in range(n_iterations):
        v_mat = R @ v_mat
        if i < n_iterations - 1:
            R_accumulated = R @ R_accumulated
    
    # expected result
    total_angle = n_iterations * angle
    R_expected = rotation_matrix_from_axis_angle(axis, total_angle)
    v_expected = R_expected @ v0
    
    # Compute errors
    error_quat = np.linalg.norm(v_quat - v_expected)
    error_mat = np.linalg.norm(v_mat - v_expected)
    orthogonality_error = check_orthogonality(R_accumulated)
    
    print(f"\nRotasi: {test_case['angle_deg']:.1f}° dengan axis {axis / np.linalg.norm(axis)}")
    print(f"Jumlah rotasi: {n_iterations}")
    
    # Output posisi vektor
    print(f"\n" + "-" * 50)
    print("VECTOR POSITIONS:")
    print("-" * 50)
    print(f"  Initial:    [{v0[0]:12.8f}, {v0[1]:12.8f}, {v0[2]:12.8f}]")
    print(f"  Quaternion: [{v_quat[0]:12.8f}, {v_quat[1]:12.8f}, {v_quat[2]:12.8f}]")
    print(f"  Matrix:     [{v_mat[0]:12.8f}, {v_mat[1]:12.8f}, {v_mat[2]:12.8f}]")
    print(f"  Expected:   [{v_expected[0]:12.8f}, {v_expected[1]:12.8f}, {v_expected[2]:12.8f}]")
    
    print(f"\nError di final vector position:")
    print(f"  Quaternion method:      {error_quat:.2e}")
    print(f"  Rotation matrix method: {error_mat:.2e}")
    if error_quat > 0:
        print(f"  Accuracy ratio:         {error_mat / error_quat:.2f}x")
    print(f"\nOrthogonality deviation:")
    print(f"  Rotation matrix method: {orthogonality_error:.2e}")
    
    return {
        'error_quat': error_quat,
        'error_mat': error_mat,
        'orthogonality': orthogonality_error
    }

def experiment_computational_cost(test_case):
    """Membandingkan execution time"""
    n_rotations = test_case['n_rotations_cost']
    
    print("\n" + "=" * 70)
    print("COMPUTATIONAL COST OF ROTATION COMPOSITION")
    print("=" * 70)
    
    # Generate random rotations
    np.random.seed(42)
    axes = [np.random.randn(3) for _ in range(n_rotations)]
    angles = [np.random.uniform(0, 2 * np.pi) for _ in range(n_rotations)]
    
    # compute quaternion
    start = perf_counter()
    q_result = Quaternion(1, 0, 0, 0)  
    for axis, angle in zip(axes, angles):
        q = Quaternion.from_axis_angle(axis, angle)
        q_result = q_result.multiply(q)
    time_quat = perf_counter() - start
    
    # compute rotation matrix
    start = perf_counter()
    R_result = np.eye(3)
    for axis, angle in zip(axes, angles):
        R = rotation_matrix_from_axis_angle(axis, angle)
        R_result = R_result @ R
    time_mat = perf_counter() - start
    
    print(f"\nJumlah rotasi: {n_rotations}")
    print(f"\nExecution time:")
    print(f"  Quaternion method: {time_quat * 1000:.2f} ms")
    print(f"  Matrix method:     {time_mat * 1000:.2f} ms")
    print(f"  Speedup factor:    {time_mat / time_quat:.2f}x")
    
    return {
        'time_quat': time_quat,
        'time_mat': time_mat,
        'speedup': time_mat / time_quat
    }

current_test = TEST_CASES[ACTIVE_TEST_CASE]
print(f"\n{'-' * 70}")
print(f"TEST CASE {ACTIVE_TEST_CASE}: {current_test['name']}")
print(f"{'-' * 70}\n")

result1 = experiment_numerical_stability(current_test)
result2 = experiment_computational_cost(current_test)

print("\n" + "=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)

if result1['error_quat'] < result1['error_mat'] and result1['error_quat'] > 0:
    stability_statement = f"Quaternions {result1['error_mat'] / result1['error_quat']:.2f}x lebih akurat"
else:
    stability_statement = f"Kedua metode memiliki akurasi sebanding di level ~{result1['error_quat']:.0e}"

print(f"\n1. Numerical Stability:")
print(f"   {stability_statement}")
print(f"   Kehilangan orthogonalitas matriks: {result1['orthogonality']:.2e}")
print(f"\n2. Computational Efficiency:")
print(f"   Quaternions {result2['speedup']:.2f}x lebih cepat")