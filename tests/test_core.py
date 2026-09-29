import unittest
import warnings

import torch

import adepts as ad


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.mesh = ad.CartesianMesh(1.0, 1.0, 0.25, 0.25, device="cpu")
        self.state = self.mesh.to_mesh_state()

    def test_mesh_layout(self):
        self.assertEqual((self.mesh.Nx, self.mesh.Ny), (5, 5))
        self.assertEqual(tuple(self.state["point_array"].shape), (6, 6, 4))

    def test_constant_interpolation(self):
        grids = {
            "node": (self.state["ynode"], self.state["xnode"]),
            "vx": (self.state["yvx"], self.state["xvx"]),
            "vy": (self.state["yvy"], self.state["xvy"]),
            "p": (self.state["yp"], self.state["xp"]),
        }
        for src, (sy, sx) in grids.items():
            values = torch.full((sy.numel(), sx.numel()), 2.5)
            for target, (ty, tx) in grids.items():
                with self.subTest(src=src, target=target):
                    out = ad.interp_between_grids(
                        self.state, values, src=src, tgt=target
                    )
                    self.assertEqual(tuple(out.shape), (ty.numel(), tx.numel()))
                    torch.testing.assert_close(out, torch.full_like(out, 2.5))

    def test_sparse_solve_gradient(self):
        crow = torch.tensor([0, 2, 4], dtype=torch.int64)
        col = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        values = torch.tensor([4.0, 1.0, 1.0, 3.0], requires_grad=True)
        matrix = torch.sparse_csr_tensor(crow, col, values, size=(2, 2))
        rhs = torch.tensor([1.0, 2.0], requires_grad=True)

        solution = ad.solvers.sparse_solve(matrix, rhs, matrix_kind="unit_test")
        torch.testing.assert_close(solution, torch.linalg.solve(matrix.to_dense(), rhs))

        solution.square().sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.isfinite(rhs.grad).all())

    def test_solver_cache_separates_matrix_sizes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for size in (2, 3):
                matrix = torch.eye(size).to_sparse_csr()
                rhs = torch.ones(size)
                solution = ad.solvers.sparse_solve(
                    matrix, rhs, matrix_kind="size_regression"
                )
                torch.testing.assert_close(solution, rhs)


if __name__ == "__main__":
    unittest.main()
