import numpy as np
import gudhi as gd

def compute_persistent_homology(series: np.ndarray, max_dim=1):
    points = np.array([[i, s] for i, s in enumerate(series)])
    rips_complex = gd.RipsComplex(points=points, max_edge_length=2.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    persistence = simplex_tree.persistence()
    return persistence
