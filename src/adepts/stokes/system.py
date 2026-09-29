import torch
from .. import interpolation
from .. import solvers
from .. import log
from .assembly import stokes_assembly
import warnings
import time
logger = log.LogManager()

class StokesBoundary:
    _enum = {
        "free_slip": -1,
        "no_slip": 0,
    }

    def __init__(self, boundary_type="free_slip"):
        self._boundary_dict = None
        self.boundary_const = None
        self.initialize(boundary_type)

    def initialize(self, boundary_type):
        if isinstance(boundary_type, str):
            bt = boundary_type.lower()
            assert bt in self._enum, f"Unknown Stokes boundary condition '{bt}'."
            self._boundary_dict = dict(left=bt, right=bt, top=bt, bottom=bt)

        elif isinstance(boundary_type, dict):
            bd = {}
            for side in ["left", "right", "top", "bottom"]:
                v = boundary_type.get(side, "free_slip").lower()
                assert v in self._enum, f"Unknown Stokes boundary condition '{v}'."
                bd[side] = v
            self._boundary_dict = bd
        else:
            raise ValueError("boundary_type must be a string or dict")

        self._update_tensor()

    def _update_tensor(self):
        self.boundary_const = (
            self._enum[self._boundary_dict["left"]],
            self._enum[self._boundary_dict["right"]],
            self._enum[self._boundary_dict["top"]],
            self._enum[self._boundary_dict["bottom"]],
        )


class StokesSystem:
    @staticmethod
    def normalize_stokes_boundary(boundary):
        enum = {
            "free_slip": -1,
            "no_slip": 0,
        }

        if isinstance(boundary, (tuple, list)):
            assert len(boundary) == 4
            return tuple(int(x) for x in boundary)

        if isinstance(boundary, torch.Tensor):
            assert boundary.shape == (4,)
            return tuple(
                int(x) for x in boundary.detach().cpu().to(torch.int64).tolist()
            )

        if isinstance(boundary, str):
            b = boundary.lower()
            assert b in enum, f"Unknown Stokes boundary condition '{b}'."
            return (enum[b], enum[b], enum[b], enum[b])

        if isinstance(boundary, dict):
            out = []
            for side in ["left", "right", "top", "bottom"]:
                b = boundary.get(side, "free_slip").lower()
                assert b in enum, f"Unknown Stokes boundary condition '{b}' on {side}."
                out.append(enum[b])
            return tuple(out)

        raise TypeError("boundary must be str, dict, tuple/list, or torch.Tensor")

    @staticmethod
    def velocity_pressure_extract(S: torch.Tensor, pscale: torch.Tensor, mesh_state):
        Nx1 = mesh_state["Nx1"]
        Ny1 = mesh_state["Ny1"]

        S_reshaped = S.view(Nx1, Ny1, 3).permute(1, 0, 2)
        vx = S_reshaped[:, :, 0]
        vy = S_reshaped[:, :, 1]
        pr = S_reshaped[:, :, 2] * pscale
        return vx, vy, pr

    @staticmethod
    def update_pressure_ghost(pressure: torch.Tensor) -> torch.Tensor:
        """
        Fill ghost/boundary pressure values by linear extrapolation.
        """

        pr = pressure.clone()

        pr[0, 1:-1] = 2.0 * pressure[1, 1:-1] - pressure[2, 1:-1]
        pr[-1, 1:-1] = 2.0 * pressure[-2, 1:-1] - pressure[-3, 1:-1]
        pr[:, 0] = 2.0 * pressure[:, 1] - pressure[:, 2]
        pr[:, -1] = 2.0 * pressure[:, -2] - pressure[:, -3]

        pr[0, 0] = 2.0 * pressure[1, 1] - pressure[2, 2]
        pr[0, -1] = 2.0 * pressure[1, -2] - pressure[2, -3]
        pr[-1, 0] = 2.0 * pressure[-2, 1] - pressure[-3, 2]
        pr[-1, -1] = 2.0 * pressure[-2, -2] - pressure[-3, -3]

        return pr

    @staticmethod
    def strain_II_node_fn(vx, vy, mesh_state):
        """
        Compute the second invariant of the strain-rate tensor on the requested
        staggered grid.
        """

        xnode = mesh_state["xnode"]
        ynode = mesh_state["ynode"]
        xvy = mesh_state["xvy"]
        yvx = mesh_state["yvx"]

        Ny = mesh_state["Ny"]
        Nx = mesh_state["Nx"]

        exy_node = 0.5 * (
            (vx[1:, :-1] - vx[:-1, :-1]) / (yvx[1:] - yvx[:-1]).unsqueeze(1)
            + (vy[:-1, 1:] - vy[:-1, :-1]) / (xvy[1:] - xvy[:-1]).unsqueeze(0)
        )

        vx_node = interpolation.interp_between_grids(
            mesh_state, vx, src="vx", tgt="node"
        )
        vy_node = interpolation.interp_between_grids(
            mesh_state, vy, src="vy", tgt="node"
        )

        exx_node = torch.zeros((Ny, Nx), dtype=vx.dtype, device=vx.device)
        eyy_node = torch.zeros((Ny, Nx), dtype=vx.dtype, device=vx.device)

        if Nx >= 3:
            exx_node[:, 1:-1] = (
                vx_node[:, 2:] - vx_node[:, :-2]
            ) / (xnode[2:] - xnode[:-2]).unsqueeze(0)

        if Nx >= 2:
            exx_node[:, 0] = (
                vx_node[:, 1] - vx_node[:, 0]
            ) / (xnode[1] - xnode[0])

            exx_node[:, -1] = (
                vx_node[:, -1] - vx_node[:, -2]
            ) / (xnode[-1] - xnode[-2])
        elif Nx == 1:
            exx_node[:, 0] = 0.0

        if Ny >= 3:
            eyy_node[1:-1, :] = (
                vy_node[2:, :] - vy_node[:-2, :]
            ) / (ynode[2:] - ynode[:-2]).unsqueeze(1)

        if Ny >= 2:
            eyy_node[0, :] = (
                vy_node[1, :] - vy_node[0, :]
            ) / (ynode[1] - ynode[0])

            eyy_node[-1, :] = (
                vy_node[-1, :] - vy_node[-2, :]
            ) / (ynode[-1] - ynode[-2])
        elif Ny == 1:
            eyy_node[0, :] = 0.0

        strain_II_node = torch.sqrt(
            0.5 * (exx_node**2 + eyy_node**2 + 2.0 * exy_node**2)
        )
        strain_II_node = torch.clamp(strain_II_node, min=1e-20)

        return strain_II_node, exx_node, eyy_node, exy_node, vx_node, vy_node

    @staticmethod
    def strain_II_p_fn(vx, vy, mesh_state):
        """
        Compute the second invariant of the strain-rate tensor on the requested
        staggered grid.
        """

        Ny1, Nx1 = mesh_state["Ny1"], mesh_state["Nx1"]
        xvx = mesh_state["xvx"]
        yvx = mesh_state["yvx"]
        xvy = mesh_state["xvy"]
        yvy = mesh_state["yvy"]

        exx_p = torch.zeros((Ny1, Nx1), dtype=vx.dtype, device=vx.device)
        eyy_p = torch.zeros((Ny1, Nx1), dtype=vx.dtype, device=vx.device)

        exy_node = 0.5 * (
            (vx[1:, :-1] - vx[:-1, :-1]) / (yvx[1:] - yvx[:-1]).unsqueeze(1)
            + (vy[:-1, 1:] - vy[:-1, :-1]) / (xvy[1:] - xvy[:-1]).unsqueeze(0)
        )

        exx_p[1:-1, 1:-1] = (
            vx[1:-1, 1:-1] - vx[1:-1, :-2]
        ) / (xvx[1:-1] - xvx[:-2]).unsqueeze(0)

        eyy_p[1:-1, 1:-1] = (
            vy[1:-1, 1:-1] - vy[:-2, 1:-1]
        ) / (yvy[1:-1] - yvy[:-2]).unsqueeze(1)

        # Boundary copies are used only for full-grid rheology evaluation.

        exx_p[0, 1:-1] = exx_p[1, 1:-1]
        exx_p[-1, 1:-1] = exx_p[-2, 1:-1]
        exx_p[:, 0] = exx_p[:, 1]
        exx_p[:, -1] = exx_p[:, -2]

        eyy_p[0, 1:-1] = eyy_p[1, 1:-1]
        eyy_p[-1, 1:-1] = eyy_p[-2, 1:-1]
        eyy_p[:, 0] = eyy_p[:, 1]
        eyy_p[:, -1] = eyy_p[:, -2]

        exx_p[0, 0] = exx_p[1, 1]
        exx_p[0, -1] = exx_p[1, -2]
        exx_p[-1, 0] = exx_p[-2, 1]
        exx_p[-1, -1] = exx_p[-2, -2]

        eyy_p[0, 0] = eyy_p[1, 1]
        eyy_p[0, -1] = eyy_p[1, -2]
        eyy_p[-1, 0] = eyy_p[-2, 1]
        eyy_p[-1, -1] = eyy_p[-2, -2]

        exy_p = interpolation.interp_between_grids(
            mesh_state, exy_node, src="node", tgt="p"
        )

        strain_II_p = torch.sqrt(
            0.5 * (exx_p**2 + eyy_p**2 + 2.0 * exy_p**2)
        )
        strain_II_p = torch.clamp(strain_II_p, min=1e-20)

        return strain_II_p, exx_p, eyy_p, exy_p, exy_node

    @staticmethod
    def linear_solve(
        dt,
        viscosity_node,
        viscosity_p,
        density_x,
        density_y,
        mesh_state,
        gx,
        gy,
        pscale,
        boundary_const,
        timestep,
        is_stick_air=False,
        p_fix_value=None,
    ):
        boundary_const = StokesSystem.normalize_stokes_boundary(boundary_const)

        if p_fix_value is None:
            p_fix_value = torch.tensor(0.0, dtype=pscale.dtype, device=pscale.device)

        row, col, val, b0 = stokes_assembly(
            viscosity_node,
            viscosity_p,
            density_x,
            density_y,
            dt,
            pscale,
            mesh_state,
            gx,
            gy,
            boundary_const,
            is_stick_air = is_stick_air,
            p_fix_value=p_fix_value,
        )

        Nx1, Ny1 = mesh_state["Nx1"], mesh_state["Ny1"]
        N = Nx1 * Ny1 * 3

        pattern = solvers.make_csr_pattern_from_coo(row, col, N)
        val_csr = solvers.coalesce_values_to_csr(val, pattern)

        u = solvers.csr_solve_from_values(
            crow=pattern.crow,
            col=pattern.col,
            row_of_val=pattern.row_of_val,
            val_csr=val_csr,
            b=b0,
            shape=pattern.shape,
            matrix_key="stokes",
            reuse_mode="structure_fixed",
            transpose=False,
        )

        # Diagnostic only; excluded from higher-order differentiation.
        A_debug = torch.sparse_csr_tensor(
            pattern.crow.to(device=val.device),
            pattern.col.to(device=val.device),
            val_csr.detach(),
            size=pattern.shape,
            device=val.device,
            dtype=val.dtype,
        )

        return u, A_debug, b0


    @staticmethod
    def nonlinear_solve(
        mesh_state,
        density_x,
        density_y,
        tkp,
        gx,
        gy,
        dt,
        pscale,
        boundary_const,
        timestep,
        *,
        viscosity_update_fn,
        rheology_context=None,
        p_fix_value=None,
        is_stick_air=False,
        u0=None,
        max_step=101,
        rel_tol=1e-4,
        nl_tol=1e-8,
        omega=1.0,
        min_omega=0.1,
        use_eta_convergence=False,
        epsII_ref_nd=None,
        fix_nonlinear_iteration=None,
        debug_print = False,
    ):
        """Solve the nonlinear Stokes system by Picard iteration."""
        if viscosity_update_fn is None:
            raise ValueError("viscosity_update_fn must be provided.")

        if p_fix_value is None:
            p_fix_value = torch.tensor(0.0, dtype=pscale.dtype, device=pscale.device)

        eps = 1e-25

        if epsII_ref_nd is None:
            if rheology_context is not None and ("epsII_ref_nd" in rheology_context):
                epsII_ref_nd = rheology_context["epsII_ref_nd"]
            elif rheology_context is not None and ("t0" in rheology_context):
                epsII_ref_nd = 1e-15 * rheology_context["t0"]
            else:
                epsII_ref_nd = torch.tensor(1e-15, dtype=tkp.dtype, device=tkp.device)

        if not torch.is_tensor(epsII_ref_nd):
            epsII_ref_nd = torch.tensor(epsII_ref_nd, dtype=tkp.dtype, device=tkp.device)
        else:
            epsII_ref_nd = epsII_ref_nd.to(dtype=tkp.dtype, device=tkp.device)

        def _build_viscosity(strain_II_p, strain_II_node, pressure_ext):
            viscosity_node, viscosity_p = viscosity_update_fn(
                mesh_state,
                strain_II_p,
                strain_II_node,
                pressure_ext,
                tkp,
                rheology_context,
            )

            if not torch.isfinite(viscosity_node).all():
                raise RuntimeError("viscosity_node contains NaN/Inf")
            if not torch.isfinite(viscosity_p).all():
                raise RuntimeError("viscosity_p contains NaN/Inf")

            expect_node = (mesh_state["Ny"], mesh_state["Nx"])
            expect_p = (mesh_state["Ny1"], mesh_state["Nx1"])

            if tuple(viscosity_node.shape) != expect_node:
                raise RuntimeError(
                    f"viscosity_node shape mismatch: got {tuple(viscosity_node.shape)}, expected {expect_node}"
                )
            if tuple(viscosity_p.shape) != expect_p:
                raise RuntimeError(
                    f"viscosity_p shape mismatch: got {tuple(viscosity_p.shape)}, expected {expect_p}"
                )

            return viscosity_node, viscosity_p

        if u0 is None:
            strain_II_p0 = torch.full(
                (mesh_state["Ny1"], mesh_state["Nx1"]),
                float(epsII_ref_nd.item()) if epsII_ref_nd.numel() == 1 else 0.0,
                dtype=tkp.dtype,
                device=tkp.device,
            )
            if epsII_ref_nd.numel() != 1:
                strain_II_p0[:] = epsII_ref_nd

            strain_II_node0 = torch.full(
                (mesh_state["Ny"], mesh_state["Nx"]),
                float(epsII_ref_nd.item()) if epsII_ref_nd.numel() == 1 else 0.0,
                dtype=tkp.dtype,
                device=tkp.device,
            )
            if epsII_ref_nd.numel() != 1:
                strain_II_node0[:] = epsII_ref_nd

            pressure_ext0 = torch.zeros_like(tkp)

            viscosity_node, viscosity_p = _build_viscosity(
                strain_II_p0, strain_II_node0, pressure_ext0
            )

            u0, A0, b0 = StokesSystem.linear_solve(
                dt,
                viscosity_node,
                viscosity_p,
                density_x,
                density_y,
                mesh_state,
                gx,
                gy,
                pscale,
                boundary_const,
                timestep,
                is_stick_air=is_stick_air,
                p_fix_value=p_fix_value,
            )
        else:
            vx0, vy0, pr0 = StokesSystem.velocity_pressure_extract(u0, pscale, mesh_state)

            strain_II_p0, *_ = StokesSystem.strain_II_p_fn(vx0, vy0, mesh_state)
            strain_II_node0, *_ = StokesSystem.strain_II_node_fn(vx0, vy0, mesh_state)
            pressure_ext0 = StokesSystem.update_pressure_ghost(pr0)

            viscosity_node, viscosity_p = _build_viscosity(
                strain_II_p0, strain_II_node0, pressure_ext0
            )

        # Extra first-step Picard iterations provide the initial warm start.
        if fix_nonlinear_iteration is not None:
            if timestep == 0 :
                max_step_eff = 20 + int(fix_nonlinear_iteration)
            else:
                max_step_eff = int(fix_nonlinear_iteration)
            if max_step_eff <= 0:
                raise ValueError("fix_nonlinear_iteration must be positive or None.")
            fixed_iter_mode = True
        else:
            max_step_eff = int(max_step)
            fixed_iter_mode = False

        prev_rel_u = None
        omega_eff = omega

        viscosity_node_old = viscosity_node.detach().clone()
        viscosity_p_old = viscosity_p.detach().clone()

        A = None
        b = None
        rel_u = None
        rel_nl = None
        rel_eta = torch.tensor(0.0, dtype=tkp.dtype, device=tkp.device)

        time0 = time.time()
        for i in range(max_step_eff):
            vx, vy, pressure = StokesSystem.velocity_pressure_extract(u0, pscale, mesh_state)
            strain_II_p, *_ = StokesSystem.strain_II_p_fn(vx, vy, mesh_state)
            strain_II_node, *_ = StokesSystem.strain_II_node_fn(vx, vy, mesh_state)
            pressure_ext = StokesSystem.update_pressure_ghost(pressure)

            viscosity_node, viscosity_p = _build_viscosity(
                strain_II_p, strain_II_node, pressure_ext
            )

            if use_eta_convergence:
                rel_eta_node = torch.norm(viscosity_node - viscosity_node_old) / (torch.norm(viscosity_node) + eps)
                rel_eta_p = torch.norm(viscosity_p - viscosity_p_old) / (torch.norm(viscosity_p) + eps)
                rel_eta = torch.max(torch.stack([rel_eta_node, rel_eta_p]))

            u_star, A, b = StokesSystem.linear_solve(
                dt,
                viscosity_node,
                viscosity_p,
                density_x,
                density_y,
                mesh_state,
                gx,
                gy,
                pscale,
                boundary_const,
                timestep,
                is_stick_air=is_stick_air,
                p_fix_value=p_fix_value,
            )

            g = u_star - u0

            u_new = u0 + omega_eff * g

            du = u_new - u0
            dvx, dvy, dpr = StokesSystem.velocity_pressure_extract(du, pscale, mesh_state)
            vx1, vy1, pr1 = StokesSystem.velocity_pressure_extract(u_new, pscale, mesh_state)

            rel_vx = torch.norm(dvx) / (torch.norm(vx1) + eps)
            rel_vy = torch.norm(dvy) / (torch.norm(vy1) + eps)
            rel_pr = torch.norm(dpr) / (torch.norm(pr1) + eps)
            rel_u = torch.max(torch.stack([rel_vx, rel_vy, rel_pr]))

            defect = A @ u0 - b
            rel_nl = torch.norm(defect) / (torch.norm(b) + eps)

            if not fixed_iter_mode:
                if prev_rel_u is not None and rel_u > prev_rel_u * 1.05:
                    omega_eff = max(min_omega, 0.5 * omega_eff)

            if fixed_iter_mode:
                converged = (i == max_step_eff - 1)

                if converged:
                    timeend = time.time()
                    picard_solve_time = timeend - time0
                    logger.log_body(
                        f"[FixedIterMode] Finished fixed nonlinear iterations: "
                        f"timestep={timestep}, "
                        f"picard_solve_time={picard_solve_time:.3f}s, "
                        f"i={i}, max_step_eff={max_step_eff}, "
                        f"rel_u={float(rel_u.detach().cpu()):.3e}, "
                        f"rel_tol={rel_tol:.3e}, "
                        f"rel_nl={float(rel_nl.detach().cpu()):.3e}, "
                        f"nl_tol={nl_tol:.3e}"
                    )


                if converged and not ((rel_u < rel_tol) or (rel_nl < nl_tol)) and (debug_print):
                    warnings.warn(
                        f"[FixedIterMode] Reached fixed iteration limit "
                        f"i={i}, max_step_eff={max_step_eff}, "
                        f"but nonlinear convergence criteria were not satisfied: "
                        f"rel_u={rel_u:.3e} >= rel_tol={rel_tol:.3e}, "
                        f"rel_nl={rel_nl:.3e} >= nl_tol={nl_tol:.3e}",
                        RuntimeWarning
                    )

                    logger.log_body(
                        f"[FixedIterMode] Reached fixed iteration limit "
                        f"i={i}, max_step_eff={max_step_eff}, "
                        f"but nonlinear convergence criteria were not satisfied: "
                        f"rel_u={rel_u:.3e} >= rel_tol={rel_tol:.3e}, "
                        f"rel_nl={rel_nl:.3e} >= nl_tol={nl_tol:.3e}",
                        RuntimeWarning
                    )

            else:
                if use_eta_convergence:
                    converged = ((rel_u < rel_tol) and (rel_eta < rel_tol)) or (rel_nl < nl_tol)
                else:
                    converged = (rel_u < rel_tol) or (rel_nl < nl_tol)

            if converged:
                if debug_print:
                    print('rel_u',rel_u)
                    print('rel_nl',rel_nl)
                    vx_defect, vy_defect, pressure_defect = StokesSystem.velocity_pressure_extract(defect, pscale, mesh_state)
                    rel_vx = 3 * torch.norm(vx_defect) / (torch.norm(b) + eps)
                    rel_vy = 3 * torch.norm(vy_defect) / (torch.norm(b) + eps)
                    print('rel_vx',rel_vx)
                    print("rel_vy",rel_vy)

                strain_II_p_final, *_ = StokesSystem.strain_II_p_fn(vx1, vy1, mesh_state)

                pressure_final = pr1
                eta_p = viscosity_p

                # Surface normal stress.
                vy_top = vy1[0, :]
                vy_1 = vy1[1, :]
                vy_2 = vy1[2, :]

                dy01 = mesh_state["yvy"][1] - mesh_state["yvy"][0]
                dy12 = mesh_state["yvy"][2] - mesh_state["yvy"][1]

                tn1_strain = -(vy_top - vy_1) / dy01
                tn2_strain = -(vy_1 - vy_2) / dy12

                eta1 = eta_p[1, :]
                eta2 = eta_p[2, :]

                p1 = pressure_final[1, :]
                p2 = pressure_final[2, :]

                tn1 = -p1 + 2.0 * eta1 * tn1_strain
                tn2 = -p2 + 2.0 * eta2 * tn2_strain

                sigma_yy_surface = 1.5 * tn1 - 0.5 * tn2
                # Remove the arbitrary pressure reference.
                sigma_yy_surface = sigma_yy_surface - sigma_yy_surface.mean()

                return {
                    "A": A,
                    "dt": dt,
                    "u": u_new,
                    "epsII_p": strain_II_p_final,
                    "viscosity_node": viscosity_node,
                    "viscosity_p": viscosity_p,
                    "sigma_yy_surface": sigma_yy_surface,
                    "pr": pr1,
                    "vx": vx1,
                    "vy": vy1,
                    'defect': defect,
                    "picard_iter": i + 1,
                    "rel_u_last": rel_u,
                    "rel_nl_last": rel_nl,
                    "rel_eta_last": rel_eta,
                    "omega_last": torch.tensor(omega_eff, dtype=tkp.dtype, device=tkp.device),
                    "converged": (not fixed_iter_mode),
                    "fixed_iter_mode": fixed_iter_mode,
                }

            u0 = u_new
            prev_rel_u = rel_u.detach()
            viscosity_node_old = viscosity_node.detach().clone()
            viscosity_p_old = viscosity_p.detach().clone()

        raise RuntimeError(
            f"Stokes nonlinear solve did NOT converge in {max_step_eff} iterations. "
            f"Last rel_u={rel_u.item():.3e}, rel_nl={rel_nl.item():.3e}, "
            f"omega={omega_eff:.3e}"
            )


    @staticmethod
    def build_stokes_residual_from_u(
        u,
        mesh_state,
        density_x,
        density_y,
        tkp,
        gx,
        gy,
        dt,
        pscale,
        boundary_const,
        timestep,
        *,
        viscosity_update_fn,
        rheology_context=None,
        p_fix_value=None,
        is_stick_air=False,
        detach_density=False,
        return_sparse_tensor=True,
    ):
        if viscosity_update_fn is None:
            raise ValueError("viscosity_update_fn must be provided.")

        boundary_const = StokesSystem.normalize_stokes_boundary(boundary_const)

        if p_fix_value is None:
            p_fix_value = torch.tensor(
                0.0,
                dtype=pscale.dtype,
                device=pscale.device,
            )

        if detach_density:
            density_x = density_x.detach()
            density_y = density_y.detach()

        vx, vy, pressure = StokesSystem.velocity_pressure_extract(
            u, pscale, mesh_state
        )

        strain_II_p, *_ = StokesSystem.strain_II_p_fn(
            vx, vy, mesh_state
        )

        strain_II_node, *_ = StokesSystem.strain_II_node_fn(
            vx, vy, mesh_state
        )

        pressure_ext = StokesSystem.update_pressure_ghost(pressure)

        viscosity_node, viscosity_p = viscosity_update_fn(
            mesh_state,
            strain_II_p,
            strain_II_node,
            pressure_ext,
            tkp,
            rheology_context,
        )

        if not torch.isfinite(viscosity_node).all():
            raise RuntimeError("viscosity_node contains NaN/Inf")
        if not torch.isfinite(viscosity_p).all():
            raise RuntimeError("viscosity_p contains NaN/Inf")

        row, col, val, b = stokes_assembly(
            viscosity_node,
            viscosity_p,
            density_x,
            density_y,
            dt,
            pscale,
            mesh_state,
            gx,
            gy,
            boundary_const,
            is_stick_air = is_stick_air,
            p_fix_value=p_fix_value,
        )

        Nx1, Ny1 = mesh_state["Nx1"], mesh_state["Ny1"]
        N = Nx1 * Ny1 * 3

        pattern = solvers.make_csr_pattern_from_coo(row, col, N)
        val_csr = solvers.coalesce_values_to_csr(val, pattern)

        crow = pattern.crow.to(device=val.device)
        ccol = pattern.col.to(device=val.device)

        csr_row = torch.repeat_interleave(
            torch.arange(N, device=val.device),
            crow[1:] - crow[:-1],
        )

        Au = u.new_zeros(N)
        Au = Au.index_add(
            0,
            csr_row,
            val_csr * u[ccol],
        )

        F = Au - b

        out = {
            "u": u,
            "F": F,
            "row": row,
            "col": col,
            "val": val,
            "b": b,
            "pattern": pattern,
            "val_csr": val_csr,
            "viscosity_node": viscosity_node,
            "viscosity_p": viscosity_p,
            "strain_II_p": strain_II_p,
            "strain_II_node": strain_II_node,
            "pressure_ext": pressure_ext,
            "vx": vx,
            "vy": vy,
            "pressure": pressure,
        }

        if return_sparse_tensor:
            A_picard = torch.sparse_csr_tensor(
                pattern.crow.to(device=val.device),
                pattern.col.to(device=val.device),
                val_csr.detach(),
                size=pattern.shape,
                device=val.device,
                dtype=val.dtype,
            )
            out["A_picard"] = A_picard

        return out


    @staticmethod
    def color_columns_from_pattern(
        rows,
        cols,
        *,
        Nrow,
        Ncol,
        device=None,
    ):

        rows_cpu = rows.detach().cpu().tolist()
        cols_cpu = cols.detach().cpu().tolist()

        row_to_cols = [[] for _ in range(Nrow)]
        for r, c in zip(rows_cpu, cols_cpu):
            row_to_cols[r].append(c)

        conflicts = [set() for _ in range(Ncol)]

        for cs in row_to_cols:
            if len(cs) <= 1:
                continue

            for c in cs:
                conflicts[c].update(cs)

            for c in cs:
                conflicts[c].discard(c)

        degrees = [len(s) for s in conflicts]

        colors_list = [-1] * Ncol
        uncolored = set(range(Ncol))
        neighbor_colors = [set() for _ in range(Ncol)]

        for _ in range(Ncol):
            j = max(
                uncolored,
                key=lambda x: (len(neighbor_colors[x]), degrees[x])
            )

            used = neighbor_colors[j]
            color = 0
            while color in used:
                color += 1

            colors_list[j] = color
            uncolored.remove(j)

            for nb in conflicts[j]:
                if nb in uncolored:
                    neighbor_colors[nb].add(color)

        colors = torch.tensor(colors_list, dtype=torch.long)

        if device is not None:
            colors = colors.to(device)

        num_colors = int(colors.max().item()) + 1

        return colors, num_colors

    @staticmethod
    def build_Fu_coloring_from_mesh(
        mesh_state,
        *,
        mode="safe",
        device=None,
        debug_print=False,
    ):
        Nx1 = int(mesh_state["Nx1"])
        Ny1 = int(mesh_state["Ny1"])

        N = Nx1 * Ny1 * 3
        s = 3 * Ny1

        offsets = set()

        if mode == "safe":
            local_range = range(-5, 6)
            one_range = range(-5, 6)
            two_range = range(-2, 3)

        elif mode == "medium":
            local_range = range(-5, 6)
            one_range = range(-5, 5)
            two_range = range(-2, 2)

        elif mode == "aggressive":
            local_range = range(-5, 6)
            one_range = range(-5, 5)
            two_range = [-2, 1]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        for d in local_range:
            offsets.add(d)

        for d in one_range:
            offsets.add(s + d)
            offsets.add(-(s + d))

        for d in two_range:
            offsets.add(2 * s + d)
            offsets.add(-(2 * s + d))

        offsets = sorted(offsets)

        rows_list = []
        cols_list = []

        for r in range(N):
            for off in offsets:
                c = r + off
                if 0 <= c < N:
                    rows_list.append(r)
                    cols_list.append(c)

        rows = torch.tensor(rows_list, dtype=torch.long)
        cols = torch.tensor(cols_list, dtype=torch.long)

        if device is not None:
            rows = rows.to(device)
            cols = cols.to(device)

        colors, num_colors = StokesSystem.color_columns_from_pattern(
            rows,
            cols,
            Nrow=N,
            Ncol=N,
            device=device,
        )

        offsets_t = torch.tensor(offsets, dtype=torch.long)
        if device is not None:
            offsets_t = offsets_t.to(device)

        if debug_print:
            print("========== build_Fu_coloring_from_mesh ==========")
            print(f"mode       = {mode}")
            print(f"Nx1        = {Nx1}")
            print(f"Ny1        = {Ny1}")
            print(f"N          = {N}")
            print(f"s=3*Ny1    = {s}")
            print(f"num offsets = {len(offsets)}")
            print(f"pattern nnz = {rows.numel()}")
            print(f"num colors  = {num_colors}")
            print("offsets:")
            print(offsets)
            print("=================================================")

        return {
            "rows": rows,
            "cols": cols,
            "colors": colors,
            "num_colors": num_colors,
            "offsets": offsets_t,
            "N": N,
            "Nx1": Nx1,
            "Ny1": Ny1,
            "mode": mode,
        }

    @staticmethod
    def build_sparse_Fu_by_coloring(
        u,
        mesh_state,
        density_x,
        density_y,
        tkp,
        gx,
        gy,
        dt,
        pscale,
        boundary_const,
        timestep,
        *,
        viscosity_update_fn,
        rheology_context=None,
        p_fix_value=None,
        is_stick_air=False,
        coloring_pack=None,
        coloring_mode="safe",
        create_sparse_tensor=True,
        return_csr=True,
        chunk_size=None,
        debug_print=False,
    ):
        from torch.func import jvp, vmap

        u_ad = u.detach().clone().requires_grad_(True)
        device = u_ad.device
        dtype = u_ad.dtype

        if coloring_pack is None:
            coloring_pack = StokesSystem.build_Fu_coloring_from_mesh(
                mesh_state,
                mode=coloring_mode,
                device=device,
                debug_print=debug_print,
            )

        rows = coloring_pack["rows"].to(device=device, dtype=torch.long)
        cols = coloring_pack["cols"].to(device=device, dtype=torch.long)
        colors = coloring_pack["colors"].to(device=device, dtype=torch.long)
        num_colors = int(coloring_pack["num_colors"])
        N = int(coloring_pack["N"])

        if u_ad.numel() != N:
            raise ValueError(
                f"u.numel()={u_ad.numel()} but coloring N={N}. "
                "Check mesh_state / coloring_pack."
            )

        entry_colors = colors[cols]

        def residual_fn_for_Fu(u_in):
            pack = StokesSystem.build_stokes_residual_from_u(
                u=u_in,
                mesh_state=mesh_state,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                gx=gx,
                gy=gy,
                dt=dt,
                pscale=pscale,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_context=rheology_context,
                p_fix_value=p_fix_value,
                is_stick_air=is_stick_air,
                detach_density=False,
                return_sparse_tensor=False,
            )
            return pack["F"].reshape(-1)

        seeds = torch.zeros(
            (num_colors, N),
            dtype=dtype,
            device=device,
        )

        seeds[colors, torch.arange(N, device=device)] = 1.0

        if debug_print:
            print("========== build_sparse_Fu_by_coloring_parallel ==========")
            print(f"N            = {N}")
            print(f"pattern nnz  = {rows.numel()}")
            print(f"num_colors   = {num_colors}")
            print(f"seeds shape  = {tuple(seeds.shape)}")
            print(f"chunk_size   = {chunk_size}")
            print("Start batched colored JVP...")
            print("==========================================================")

        def one_jvp(seed):
            _, y = jvp(
                residual_fn_for_Fu,
                (u_ad,),
                (seed,),
            )
            return y.reshape(-1)

        if chunk_size is None:
            Y = vmap(one_jvp)(seeds)

            vals = Y[entry_colors, rows]

        else:
            vals = torch.empty(rows.numel(), dtype=dtype, device=device)

            for c0 in range(0, num_colors, int(chunk_size)):
                c1 = min(c0 + int(chunk_size), num_colors)

                Y_chunk = vmap(one_jvp)(seeds[c0:c1])

                mask = (entry_colors >= c0) & (entry_colors < c1)
                idx = torch.where(mask)[0]

                local_colors = entry_colors[idx] - c0
                vals[idx] = Y_chunk[local_colors, rows[idx]]

                if debug_print:
                    print(
                        f"[color chunk {c0:4d}:{c1:4d}] "
                        f"entries={idx.numel():8d}"
                    )

        out = {
            "rows": rows,
            "cols": cols,
            "vals": vals,
            "colors": colors,
            "num_colors": num_colors,
            "N": N,
            "coloring_pack": coloring_pack,
        }

        if create_sparse_tensor:
            indices = torch.stack([rows, cols], dim=0)

            Fu_coo = torch.sparse_coo_tensor(
                indices,
                vals,
                size=(N, N),
                dtype=dtype,
                device=device,
            ).coalesce()

            out["Fu_coo"] = Fu_coo

            if return_csr:
                Fu_csr = Fu_coo.to_sparse_csr()
                out["Fu_csr"] = Fu_csr

        if debug_print:
            print("========== sparse Fu parallel done ==========")
            print(f"num_colors  = {num_colors}")
            print(f"nnz pattern = {rows.numel()}")
            print(f"||vals||    = {torch.norm(vals).item():.16e}")
            if create_sparse_tensor:
                print(f"Fu_coo nnz  = {out['Fu_coo']._nnz()}")
                if return_csr:
                    print(f"Fu_csr nnz  = {out['Fu_csr'].values().numel()}")
            print("=============================================")

        return out

    @staticmethod
    def build_sparse_Fu_by_coloring_graph(
        u,
        mesh_state,
        density_x,
        density_y,
        tkp,
        gx,
        gy,
        dt,
        pscale,
        boundary_const,
        timestep,
        *,
        viscosity_update_fn,
        rheology_context=None,
        p_fix_value=None,
        is_stick_air=False,
        coloring_pack=None,
        coloring_mode="safe",
        chunk_size=None,
        debug_print=False,
    ):
        if not torch.is_tensor(u):
            raise TypeError("u must be a torch.Tensor.")

        # Keep the external autograd graph.
        if u.requires_grad:
            u_ad = u
        else:
            raise RuntimeError(
                "build_sparse_Fu_by_coloring_graph requires u.requires_grad=True. "
                "You probably passed a detached u. For second-order AD, pass u_req or save u without detach."
            )

        device = u_ad.device
        dtype = u_ad.dtype

        if coloring_pack is None:
            coloring_pack = StokesSystem.build_Fu_coloring_from_mesh(
                mesh_state,
                mode=coloring_mode,
                device=device,
                debug_print=debug_print,
            )

        rows = coloring_pack["rows"].to(device=device, dtype=torch.long)
        cols = coloring_pack["cols"].to(device=device, dtype=torch.long)
        colors = coloring_pack["colors"].to(device=device, dtype=torch.long)
        num_colors = int(coloring_pack["num_colors"])
        N = int(coloring_pack["N"])

        if u_ad.numel() != N:
            raise ValueError(
                f"u.numel()={u_ad.numel()} but coloring N={N}. "
                "Check mesh_state / coloring_pack."
            )

        entry_colors = colors[cols]

        def residual_fn_for_Fu(u_in):
            pack = StokesSystem.build_stokes_residual_from_u(
                u=u_in,
                mesh_state=mesh_state,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                gx=gx,
                gy=gy,
                dt=dt,
                pscale=pscale,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_context=rheology_context,
                p_fix_value=p_fix_value,
                is_stick_air=is_stick_air,
                detach_density=False,
                return_sparse_tensor=False,
            )
            return pack["F"].reshape(-1)

        seeds = torch.zeros(
            (num_colors, N),
            dtype=dtype,
            device=device,
        )
        seeds[colors, torch.arange(N, device=device)] = 1.0

        if debug_print:
            print("========== build_sparse_Fu_by_coloring_graph ==========")
            print(f"N            = {N}")
            print(f"pattern nnz  = {rows.numel()}")
            print(f"num_colors   = {num_colors}")
            print(f"seeds shape  = {tuple(seeds.shape)}")
            print(f"chunk_size   = {chunk_size}")
            print("Start graph-preserving colored JVP...")
            print("=======================================================")

        if chunk_size is None:
            vals_parts = []

            for c in range(num_colors):
                seed = seeds[c]

                _, y = torch.autograd.functional.jvp(
                    lambda uu: residual_fn_for_Fu(uu),
                    inputs=(u_ad,),
                    v=(seed,),
                    create_graph=True,
                    strict=False,
                )
                y = y.reshape(-1)

                idx = torch.where(entry_colors == c)[0]
                vals_c = y[rows[idx]]

                vals_parts.append((idx, vals_c))

                if debug_print:
                    print(
                        f"[graph color {c+1:4d}/{num_colors}] "
                        f"cols={(colors == c).sum().item():6d}, "
                        f"entries={idx.numel():8d}"
                    )

            # Preserve the graph during sparse reconstruction.
            vals = torch.zeros(rows.numel(), dtype=dtype, device=device)
            for idx, vals_c in vals_parts:
                vals = vals.index_copy(0, idx, vals_c)

        else:
            vals = torch.zeros(rows.numel(), dtype=dtype, device=device)

            for c0 in range(0, num_colors, int(chunk_size)):
                c1 = min(c0 + int(chunk_size), num_colors)

                idx_all = []
                val_all = []

                for c in range(c0, c1):
                    seed = seeds[c]

                    _, y = torch.autograd.functional.jvp(
                        lambda uu: residual_fn_for_Fu(uu),
                        inputs=(u_ad,),
                        v=(seed,),
                        create_graph=True,
                        strict=False,
                    )
                    y = y.reshape(-1)

                    idx = torch.where(entry_colors == c)[0]
                    vals_c = y[rows[idx]]

                    idx_all.append(idx)
                    val_all.append(vals_c)

                if len(idx_all) > 0:
                    idx_cat = torch.cat(idx_all, dim=0)
                    val_cat = torch.cat(val_all, dim=0)
                    vals = vals.index_copy(0, idx_cat, val_cat)

                if debug_print:
                    print(
                        f"[graph color chunk {c0:4d}:{c1:4d}] "
                        f"entries={sum(x.numel() for x in idx_all):8d}"
                    )

        out = {
            "rows": rows,
            "cols": cols,
            "vals": vals,
            "colors": colors,
            "num_colors": num_colors,
            "N": N,
            "coloring_pack": coloring_pack,
        }

        if debug_print:
            print("========== sparse Fu graph done ==========")
            print(f"num_colors  = {num_colors}")
            print(f"nnz pattern = {rows.numel()}")
            print(f"vals.requires_grad = {vals.requires_grad}")
            print(f"||vals||    = {torch.norm(vals).item():.16e}")
            print("==========================================")

        return out


    @staticmethod
    def picard_solve_for_implicit(
        mesh_state,
        density_x,
        density_y,
        tkp,
        gx,
        gy,
        dt,
        pscale,
        boundary_const,
        timestep,
        *,
        viscosity_update_fn,
        rheology_context=None,
        p_fix_value=None,
        is_stick_air=False,
        u0=None,
        picard_steps=20,
        rel_tol=1e-10,
        omega=1.0,
        epsII_ref_nd=None,
        debug_print=False,
    ):
        """
        Picard warm start used only by the implicit nonlinear solver.
        Convergence is measured by the true nonlinear residual:
            rel_F = ||F(u)|| / ||b||
        """
        if viscosity_update_fn is None:
            raise ValueError("viscosity_update_fn must be provided.")

        if p_fix_value is None:
            p_fix_value = torch.tensor(0.0, dtype=tkp.dtype, device=tkp.device)

        eps = torch.tensor(1e-30, dtype=tkp.dtype, device=tkp.device)

        if epsII_ref_nd is None:
            if rheology_context is not None and ("epsII_ref_nd" in rheology_context):
                epsII_ref_nd = rheology_context["epsII_ref_nd"]
            elif rheology_context is not None and ("t0" in rheology_context):
                epsII_ref_nd = 1e-15 * rheology_context["t0"]
            else:
                epsII_ref_nd = torch.tensor(1e-15, dtype=tkp.dtype, device=tkp.device)

        if not torch.is_tensor(epsII_ref_nd):
            epsII_ref_nd = torch.tensor(epsII_ref_nd, dtype=tkp.dtype, device=tkp.device)
        else:
            epsII_ref_nd = epsII_ref_nd.to(dtype=tkp.dtype, device=tkp.device)

        def build_initial_u():
            if u0 is not None:
                return u0.detach()

            strain_II_p0 = torch.full(
                (mesh_state["Ny1"], mesh_state["Nx1"]),
                float(epsII_ref_nd.item()) if epsII_ref_nd.numel() == 1 else 0.0,
                dtype=tkp.dtype,
                device=tkp.device,
            )

            strain_II_node0 = torch.full(
                (mesh_state["Ny"], mesh_state["Nx"]),
                float(epsII_ref_nd.item()) if epsII_ref_nd.numel() == 1 else 0.0,
                dtype=tkp.dtype,
                device=tkp.device,
            )

            if epsII_ref_nd.numel() != 1:
                strain_II_p0[:] = epsII_ref_nd
                strain_II_node0[:] = epsII_ref_nd

            pressure_ext0 = torch.zeros_like(tkp)

            viscosity_node0, viscosity_p0 = viscosity_update_fn(
                mesh_state,
                strain_II_p0,
                strain_II_node0,
                pressure_ext0,
                tkp,
                rheology_context,
            )

            u_init, _, _ = StokesSystem.linear_solve(
                dt,
                viscosity_node0,
                viscosity_p0,
                density_x,
                density_y,
                mesh_state,
                gx,
                gy,
                pscale,
                boundary_const,
                timestep,
                is_stick_air=is_stick_air,
                p_fix_value=p_fix_value,
            )

            return u_init.detach()

        def compute_rel_F(u_in):
            pack = StokesSystem.build_stokes_residual_from_u(
                u=u_in,
                mesh_state=mesh_state,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                gx=gx,
                gy=gy,
                dt=dt,
                pscale=pscale,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_context=rheology_context,
                p_fix_value=p_fix_value,
                is_stick_air=is_stick_air,
                detach_density=False,
                return_sparse_tensor=False,
            )

            F = pack["F"]
            b = pack["b"]
            rel_F = torch.norm(F) / (torch.norm(b) + eps)
            return rel_F.detach()

        u = build_initial_u()

        # If picard_steps == 0, skip Picard and return the initial guess.
        if int(picard_steps) <= 0:
            rel_F = compute_rel_F(u)
            return {
                "u": u.detach(),
                "converged": bool(rel_F < rel_tol),
                "picard_iter": 0,
                "rel_F_last": rel_F,
                "rel_u_last": torch.full((), float("nan"), dtype=tkp.dtype, device=tkp.device),
            }

        rel_u = torch.full((), float("nan"), dtype=tkp.dtype, device=tkp.device)
        rel_F = compute_rel_F(u)

        for it in range(int(picard_steps)):
            vx, vy, pressure = StokesSystem.velocity_pressure_extract(
                u,
                pscale,
                mesh_state,
            )

            strain_II_p, *_ = StokesSystem.strain_II_p_fn(vx, vy, mesh_state)
            strain_II_node, *_ = StokesSystem.strain_II_node_fn(vx, vy, mesh_state)
            pressure_ext = StokesSystem.update_pressure_ghost(pressure)

            viscosity_node, viscosity_p = viscosity_update_fn(
                mesh_state,
                strain_II_p,
                strain_II_node,
                pressure_ext,
                tkp,
                rheology_context,
            )

            u_star, _, _ = StokesSystem.linear_solve(
                dt,
                viscosity_node,
                viscosity_p,
                density_x,
                density_y,
                mesh_state,
                gx,
                gy,
                pscale,
                boundary_const,
                timestep,
                is_stick_air=is_stick_air,
                p_fix_value=p_fix_value,
            )

            u_new = u + omega * (u_star - u)

            du = u_new - u
            dvx, dvy, dpr = StokesSystem.velocity_pressure_extract(
                du,
                pscale,
                mesh_state,
            )
            vx_new, vy_new, pr_new = StokesSystem.velocity_pressure_extract(
                u_new,
                pscale,
                mesh_state,
            )

            rel_vx = torch.norm(dvx) / (torch.norm(vx_new) + eps)
            rel_vy = torch.norm(dvy) / (torch.norm(vy_new) + eps)
            rel_pr = torch.norm(dpr) / (torch.norm(pr_new) + eps)
            rel_u = torch.max(torch.stack([rel_vx, rel_vy, rel_pr])).detach()

            u = u_new.detach()
            rel_F = compute_rel_F(u)

            if debug_print:
                print(
                    f"[Implicit-Picard] timestep={timestep}, "
                    f"iter={it + 1}, "
                    f"rel_F={rel_F.item():.6e}, "
                    f"rel_u={rel_u.item():.6e}"
                )

            if rel_F < rel_tol:
                return {
                    "u": u.detach(),
                    "converged": True,
                    "picard_iter": it + 1,
                    "rel_F_last": rel_F,
                    "rel_u_last": rel_u,
                }

        return {
            "u": u.detach(),
            "converged": False,
            "picard_iter": int(picard_steps),
            "rel_F_last": rel_F,
            "rel_u_last": rel_u,
        }

    @staticmethod
    def nonlinear_solve_fu(
        mesh_state,
        density_x,
        density_y,
        tkp,
        gx,
        gy,
        dt,
        pscale,
        boundary_const,
        timestep,
        *,
        viscosity_update_fn,
        rheology_context=None,
        p_fix_value=None,
        is_stick_air=False,
        u0=None,
        picard_steps=20,
        newton_steps=20,
        rel_tol=1e-10,
        line_search=True,
        coloring_pack=None,
        debug_print=False,
    ):
        """
        Nonlinear Stokes solve:
            1. first do a few Picard iterations
            2. then switch to exact Newton using Fu=dF/du from coloring JVP

        Solve:
            Fu(u_k) delta = -F(u_k)
            u_{k+1} = u_k + alpha * delta
        """
        def _sync_cuda_from_tensor(x):
            if torch.is_tensor(x) and x.device.type == "cuda":
                torch.cuda.synchronize(x.device)

        time_start = time.time()
        time_preprocess_total = 0.0
        time_fu_build_total = 0.0
        time_solver_total = 0.0

        time_picard_start = time.time()

        if timestep == 0:
            picard_steps = 50 + picard_steps
        picard_state = StokesSystem.picard_solve_for_implicit(
            mesh_state=mesh_state,
            density_x=density_x,
            density_y=density_y,
            tkp=tkp,
            gx=gx,
            gy=gy,
            dt=dt,
            pscale=pscale,
            boundary_const=boundary_const,
            timestep=timestep,
            viscosity_update_fn=viscosity_update_fn,
            rheology_context=rheology_context,
            p_fix_value=p_fix_value,
            is_stick_air=is_stick_air,
            u0=u0,
            picard_steps=picard_steps,
            rel_tol=rel_tol,
            omega=1.0,
            debug_print=debug_print,
        )

        time_picard_end = time.time()
        time_preprocess_total += time_picard_end - time_picard_start

        u = picard_state["u"].detach()

        if bool(picard_state["converged"]):
            vx, vy, pr = StokesSystem.velocity_pressure_extract(
                u,
                pscale,
                mesh_state,
            )

            logger.log_body(
                f"timestep={timestep}, "
                f"[Implicit Picard converged] "
                f"picard_iter={picard_state['picard_iter']}, "
                f"rel_F={picard_state['rel_F_last'].item():.6e}, "
                f"rel_u={picard_state['rel_u_last'].item():.6e}, "
                f"rel_tol={rel_tol:.6e}, "
                f"picard_time={time_preprocess_total:.2f}s."
            )

            return {
                "u": u,
                "vx": vx,
                "vy": vy,
                "pr": pr,
                "coloring_pack": coloring_pack,
                "picard_iter": picard_state["picard_iter"],
                "newton_iter": 0,
                "rel_F_last": picard_state["rel_F_last"].detach(),
                "solver_stage": "picard",
            }

        if int(newton_steps) <= 0:
            msg = (
                f"timestep={timestep}, "
                f"[Implicit nonlinear solve FAILED] "
                f"Picard did not converge after {picard_steps} steps, "
                f"newton_steps={newton_steps}. "
                f"last rel_F={picard_state['rel_F_last'].item():.6e}, "
                f"rel_tol={rel_tol:.6e}."
            )
            print(msg)
            logger.log_body(msg + "\n")
            raise ValueError(msg)

        if coloring_pack is None:
            _sync_cuda_from_tensor(u)
            time_coloring_start = time.time()

            coloring_pack = StokesSystem.build_Fu_coloring_from_mesh(
                mesh_state,
                mode="safe",
                device=u.device,
                debug_print=debug_print,
            )

            _sync_cuda_from_tensor(u)
            time_coloring_end = time.time()
            time_preprocess_total += time_coloring_end - time_coloring_start

        eps = torch.tensor(1e-30, dtype=u.dtype, device=u.device)

        converged = False

        for it in range(newton_steps):
            u_req = u.detach().clone().requires_grad_(True)

            pack_F = StokesSystem.build_stokes_residual_from_u(
                u=u_req,
                mesh_state=mesh_state,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                gx=gx,
                gy=gy,
                dt=dt,
                pscale=pscale,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_context=rheology_context,
                p_fix_value=p_fix_value,
                is_stick_air=is_stick_air,
                detach_density=False,
                return_sparse_tensor=False,
            )

            F = pack_F["F"].detach()
            b = pack_F["b"].detach()
            rel_F = torch.norm(F) / (torch.norm(b) + eps)

            if debug_print:
                print(f"[Newton-Fu] iter={it}, ||F||/||b||={rel_F.item():.6e}")

            if it >= 1:
                dvx, dvy, dp = StokesSystem.velocity_pressure_extract(
                    delta,
                    pscale,
                    mesh_state,
                )

                vx_now, vy_now, p_now = StokesSystem.velocity_pressure_extract(
                    u,
                    pscale,
                    mesh_state,
                )

                rel_dvx = torch.norm(dvx) / (torch.norm(vx_now) + eps)
                rel_dvy = torch.norm(dvy) / (torch.norm(vy_now) + eps)
                rel_dp  = torch.norm(dp)  / (torch.norm(p_now) + eps)
            else:
                rel_dvx = torch.tensor(1.0, dtype=u.dtype, device=u.device)
                rel_dvy = torch.tensor(1.0, dtype=u.dtype, device=u.device)
                rel_dp  = torch.tensor(1.0, dtype=u.dtype, device=u.device)


            if rel_F < rel_tol:
                converged = True
                _sync_cuda_from_tensor(u)
                time_end = time.time()
                logger.log_body(
                    f"timestep={timestep}, "
                    f"[Newton-Fu converged] iter={it}, "
                    f"||F||/||b||={rel_F.item():.6e}, "
                    f"rel_dvx={rel_dvx.item():.6e}, "
                    f"rel_dvy={rel_dvy.item():.6e}, "
                    f"rel_dp={rel_dp.item():.6e}, "
                    f"preprocess_time={time_preprocess_total:.2f}s, "
                    f"Fu_build_time={time_fu_build_total:.2f}s, "
                    f"solver_time={time_solver_total:.2f}s, "
                    f"nonlinear_solve_time={time_end - time_start:.2f}s."
                )

                break

            _sync_cuda_from_tensor(u)
            time_fu_start = time.time()

            fu_pack = StokesSystem.build_sparse_Fu_by_coloring(
                u=u.detach(),
                mesh_state=mesh_state,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                gx=gx,
                gy=gy,
                dt=dt,
                pscale=pscale,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_context=rheology_context,
                p_fix_value=p_fix_value,
                is_stick_air=is_stick_air,
                coloring_pack=coloring_pack,
                create_sparse_tensor=True,
                return_csr=False,
                debug_print=debug_print,
            )

            rows = fu_pack["rows"]
            cols = fu_pack["cols"]
            vals = fu_pack["vals"]

            N = int(fu_pack["N"])

            pattern = solvers.make_csr_pattern_from_coo(rows, cols, N)
            val_csr = solvers.coalesce_values_to_csr(vals, pattern)

            _sync_cuda_from_tensor(u)
            time_fu_end = time.time()
            time_fu_build_total += time_fu_end - time_fu_start

            rhs = -F

            _sync_cuda_from_tensor(u)
            time_solver_start = time.time()

            delta = solvers.csr_solve_from_values(
                crow=pattern.crow,
                col=pattern.col,
                row_of_val=pattern.row_of_val,
                val_csr=val_csr.detach(),
                b=rhs,
                shape=pattern.shape,
                matrix_key="stokes_fu_newton",
                reuse_mode="structure_fixed",
                transpose=False,
            )

            _sync_cuda_from_tensor(u)
            time_solver_end = time.time()
            time_solver_total += time_solver_end - time_solver_start


            alpha = 1.0

            if line_search:
                F_norm0 = torch.norm(F)

                accepted = False
                for _ in range(8):
                    u_trial = u + alpha * delta

                    pack_trial = StokesSystem.build_stokes_residual_from_u(
                        u=u_trial,
                        mesh_state=mesh_state,
                        density_x=density_x,
                        density_y=density_y,
                        tkp=tkp,
                        gx=gx,
                        gy=gy,
                        dt=dt,
                        pscale=pscale,
                        boundary_const=boundary_const,
                        timestep=timestep,
                        viscosity_update_fn=viscosity_update_fn,
                        rheology_context=rheology_context,
                        p_fix_value=p_fix_value,
                        is_stick_air=is_stick_air,
                        detach_density=False,
                        return_sparse_tensor=False,
                    )

                    F_trial = pack_trial["F"].detach()

                    if torch.norm(F_trial) <= 0.99 * F_norm0:
                        accepted = True
                        break

                    alpha *= 0.5

                if not accepted:
                    u_trial = u + alpha * delta
            else:
                u_trial = u + delta

            u = u_trial.detach()

        else:
            msg = (
                f"timestep={timestep}, "
                f"[Newton-Fu FAILED] "
                f"not converged after {newton_steps} steps, "
                f"last ||F||/||b||={rel_F.item():.6e}, "
                f"rel_tol={rel_tol:.6e}"
            )
            print(msg)
            logger.log_body(msg + "\n")
            raise RuntimeError(msg)

        vx, vy, pr = StokesSystem.velocity_pressure_extract(u, pscale, mesh_state)

        out = {
            "u": u,
            "vx": vx,
            "vy": vy,
            "pr": pr,
            "coloring_pack": coloring_pack,
            "newton_iter": it + 1,
            "rel_F_last": rel_F.detach(),
        }

        return out

    @staticmethod
    def implicit_solve_state(
        mesh_state,
        density_x,
        density_y,
        tkp,
        gx,
        gy,
        dt,
        pscale,
        boundary_const,
        timestep,
        *,
        viscosity_update_fn,
        rheology_context=None,
        p_fix_value=None,
        is_stick_air=False,
        u0=None,
        coloring_pack=None,
        picard_steps=10,
        newton_steps=20,
        rel_tol=1e-8,
        line_search=True,

    ):
        if rheology_context is None:
            rheology_context = {}

        if p_fix_value is None:
            p_fix_value = torch.tensor(
                0.0,
                dtype=tkp.dtype,
                device=tkp.device,
            )

        if not torch.is_tensor(gx):
            gx = torch.tensor(gx, dtype=tkp.dtype, device=tkp.device)

        if not torch.is_tensor(gy):
            gy = torch.tensor(gy, dtype=tkp.dtype, device=tkp.device)

        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, dtype=tkp.dtype, device=tkp.device)

        if not torch.is_tensor(pscale):
            pscale = torch.tensor(pscale, dtype=tkp.dtype, device=tkp.device)

        u = stokes_implicit_solve(
            mesh_state=mesh_state,
            density_x=density_x,
            density_y=density_y,
            tkp=tkp,
            gx=gx,
            gy=gy,
            dt=dt,
            pscale=pscale,
            boundary_const=boundary_const,
            timestep=timestep,
            viscosity_update_fn=viscosity_update_fn,
            rheology_context=rheology_context,
            p_fix_value=p_fix_value,
            is_stick_air=is_stick_air,
            u0=u0,
            coloring_pack=coloring_pack,
            picard_steps=picard_steps,
            newton_steps=newton_steps,
            rel_tol=rel_tol,
            line_search=line_search,
        )

        vx, vy, pr = StokesSystem.velocity_pressure_extract(
            u,
            pscale,
            mesh_state,
        )

        strain_II_p, *_ = StokesSystem.strain_II_p_fn(
            vx,
            vy,
            mesh_state,
        )

        strain_II_node, *_ = StokesSystem.strain_II_node_fn(
            vx,
            vy,
            mesh_state,
        )

        pressure_ext = StokesSystem.update_pressure_ghost(pr)

        viscosity_node, viscosity_p = viscosity_update_fn(
            mesh_state,
            strain_II_p,
            strain_II_node,
            pressure_ext,
            tkp,
            rheology_context,
        )

        eta_p = viscosity_p
        pressure_final = pr

        vy_top = vy[0, :]
        vy_1 = vy[1, :]
        vy_2 = vy[2, :]

        dy01 = mesh_state["yvy"][1] - mesh_state["yvy"][0]
        dy12 = mesh_state["yvy"][2] - mesh_state["yvy"][1]

        tn1_strain = -(vy_top - vy_1) / dy01
        tn2_strain = -(vy_1 - vy_2) / dy12

        eta1 = eta_p[1, :]
        eta2 = eta_p[2, :]

        p1 = pressure_final[1, :]
        p2 = pressure_final[2, :]

        tn1 = -p1 + 2.0 * eta1 * tn1_strain
        tn2 = -p2 + 2.0 * eta2 * tn2_strain

        sigma_yy_surface = 1.5 * tn1 - 0.5 * tn2
        sigma_yy_surface = sigma_yy_surface - sigma_yy_surface.mean()

        stokes_state = {
            "u": u,
            "vx": vx,
            "vy": vy,
            "pr": pr,
            "pressure": pr,

            "epsII_p": strain_II_p,
            "epsII_node": strain_II_node,
            "strain_II_p": strain_II_p,
            "strain_II_node": strain_II_node,

            "viscosity_node": viscosity_node,
            "viscosity_p": viscosity_p,

            "sigma_yy_surface": sigma_yy_surface,

            "dt": dt,
            "p_fix_value": p_fix_value,
            "implicit_adjoint": True,
        }

        return stokes_state





class StokesImplicitAdjointFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        density_x,
        density_y,
        tkp,
        A_param,
        n_param,
        E_param,
        V_param,
        cohesion_param,
        friction_param,
        plastic_strain_param,
        dt,
        pscale,
        gx,
        gy,
        p_fix_value,
        grad_u,
        mesh_state,
        boundary_const,
        timestep,
        viscosity_update_fn,
        rheology_static_context,
        is_stick_air,
        coloring_pack,
    ):
        ctx.save_for_backward(
            u,
            density_x,
            density_y,
            tkp,
            A_param,
            n_param,
            E_param,
            V_param,
            cohesion_param,
            friction_param,
            plastic_strain_param,
            dt,
            pscale,
            gx,
            gy,
            p_fix_value,
            grad_u,
        )

        ctx.mesh_state = mesh_state
        ctx.boundary_const = boundary_const
        ctx.timestep = timestep
        ctx.viscosity_update_fn = viscosity_update_fn
        ctx.rheology_static_context = rheology_static_context
        ctx.is_stick_air = is_stick_air
        ctx.coloring_pack = coloring_pack

        with torch.no_grad():
            grad_tuple = StokesImplicitSolveFn._implicit_adjoint_eval_numeric(
                u=u,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                A_param=A_param,
                n_param=n_param,
                E_param=E_param,
                V_param=V_param,
                cohesion_param=cohesion_param,
                friction_param=friction_param,
                plastic_strain_param=plastic_strain_param,
                dt=dt,
                pscale=pscale,
                gx=gx,
                gy=gy,
                p_fix_value=p_fix_value,
                grad_u=grad_u,
                mesh_state=mesh_state,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_static_context=rheology_static_context,
                is_stick_air=is_stick_air,
                coloring_pack=coloring_pack,
            )

        return grad_tuple

    @staticmethod
    def backward(
        ctx,
        cot_grad_density_x,
        cot_grad_density_y,
        cot_grad_tkp,
        cot_grad_A,
        cot_grad_n,
        cot_grad_E,
        cot_grad_V,
        cot_grad_cohesion,
        cot_grad_friction,
        cot_grad_plastic_strain,
        cot_grad_dt,
        cot_grad_pscale,
        cot_grad_gx,
        cot_grad_gy,
        cot_grad_p_fix,
    ):
        (
            u,
            density_x,
            density_y,
            tkp,
            A_param,
            n_param,
            E_param,
            V_param,
            cohesion_param,
            friction_param,
            plastic_strain_param,
            dt,
            pscale,
            gx,
            gy,
            p_fix_value,
            grad_u,
        ) = ctx.saved_tensors

        mesh_state = ctx.mesh_state
        boundary_const = ctx.boundary_const
        timestep = ctx.timestep
        viscosity_update_fn = ctx.viscosity_update_fn
        rheology_static_context = ctx.rheology_static_context
        is_stick_air = ctx.is_stick_air
        coloring_pack = ctx.coloring_pack

        with torch.enable_grad():
            u_req = u.detach().clone().requires_grad_(True)

            density_x_req = density_x.detach().clone().requires_grad_(True)
            density_y_req = density_y.detach().clone().requires_grad_(True)
            tkp_req = tkp.detach().clone().requires_grad_(True)

            A_req = A_param.detach().clone().requires_grad_(True)
            n_req = n_param.detach().clone().requires_grad_(True)
            E_req = E_param.detach().clone().requires_grad_(True)
            V_req = V_param.detach().clone().requires_grad_(True)

            cohesion_req = cohesion_param.detach().clone().requires_grad_(True)
            friction_req = friction_param.detach().clone().requires_grad_(True)
            plastic_strain_req = plastic_strain_param.detach().clone().requires_grad_(True)

            dt_req = dt.detach().clone().requires_grad_(True)
            pscale_req = pscale.detach().clone().requires_grad_(True)
            gx_req = gx.detach().clone().requires_grad_(True)
            gy_req = gy.detach().clone().requires_grad_(True)
            p_fix_req = p_fix_value.detach().clone().requires_grad_(True)

            grad_u_req = grad_u.detach().clone().requires_grad_(True)

            grad_tuple_graph = StokesImplicitSolveFn._implicit_adjoint_eval_graph(
                u=u_req,
                density_x=density_x_req,
                density_y=density_y_req,
                tkp=tkp_req,
                A_param=A_req,
                n_param=n_req,
                E_param=E_req,
                V_param=V_req,
                cohesion_param=cohesion_req,
                friction_param=friction_req,
                plastic_strain_param=plastic_strain_req,
                dt=dt_req,
                pscale=pscale_req,
                gx=gx_req,
                gy=gy_req,
                p_fix_value=p_fix_req,
                grad_u=grad_u_req,
                mesh_state=mesh_state,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_static_context=rheology_static_context,
                is_stick_air=is_stick_air,
                coloring_pack=coloring_pack,
            )

            cot_tuple = (
                cot_grad_density_x,
                cot_grad_density_y,
                cot_grad_tkp,
                cot_grad_A,
                cot_grad_n,
                cot_grad_E,
                cot_grad_V,
                cot_grad_cohesion,
                cot_grad_friction,
                cot_grad_plastic_strain,
                cot_grad_dt,
                cot_grad_pscale,
                cot_grad_gx,
                cot_grad_gy,
                cot_grad_p_fix,
            )

            phi = None
            for g_i, c_i in zip(grad_tuple_graph, cot_tuple):
                if g_i is None or c_i is None:
                    continue
                term = torch.sum(g_i * c_i)
                phi = term if phi is None else phi + term

            if phi is None:
                phi = torch.zeros((), dtype=tkp.dtype, device=tkp.device)

            inputs = [
                u_req,
                density_x_req,
                density_y_req,
                tkp_req,
                A_req,
                n_req,
                E_req,
                V_req,
                cohesion_req,
                friction_req,
                plastic_strain_req,
                dt_req,
                pscale_req,
                gx_req,
                gy_req,
                p_fix_req,
                grad_u_req,
            ]

            grads = torch.autograd.grad(
                phi,
                inputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        (
            grad_u_input,
            grad_density_x_input,
            grad_density_y_input,
            grad_tkp_input,
            grad_A_input,
            grad_n_input,
            grad_E_input,
            grad_V_input,
            grad_cohesion_input,
            grad_friction_input,
            grad_plastic_strain_input,
            grad_dt_input,
            grad_pscale_input,
            grad_gx_input,
            grad_gy_input,
            grad_p_fix_input,
            grad_grad_u_input,
        ) = grads

        return (
            grad_u_input,
            grad_density_x_input,
            grad_density_y_input,
            grad_tkp_input,
            grad_A_input,
            grad_n_input,
            grad_E_input,
            grad_V_input,
            grad_cohesion_input,
            grad_friction_input,
            grad_plastic_strain_input,
            grad_dt_input,
            grad_pscale_input,
            grad_gx_input,
            grad_gy_input,
            grad_p_fix_input,
            grad_grad_u_input,
            None,  # mesh_state
            None,  # boundary_const
            None,  # timestep
            None,  # viscosity_update_fn
            None,  # rheology_static_context
            None,  # is_stick_air
            None,  # coloring_pack
        )

class StokesImplicitSolveFn(torch.autograd.Function):
    @staticmethod
    def _implicit_adjoint_eval_numeric(
        *,
        u,
        density_x,
        density_y,
        tkp,
        A_param,
        n_param,
        E_param,
        V_param,
        cohesion_param,
        friction_param,
        plastic_strain_param,
        dt,
        pscale,
        gx,
        gy,
        p_fix_value,
        grad_u,
        mesh_state,
        boundary_const,
        timestep,
        viscosity_update_fn,
        rheology_static_context,
        is_stick_air,
        coloring_pack,
    ):

        if rheology_static_context is None:
            rheology_static_context = {}

        # First-order adjoint.
        rheology_context_for_Fu = dict(rheology_static_context)
        rheology_context_for_Fu.update(
            {
                "A_param": A_param.detach(),
                "n_param": n_param.detach(),
                "E_param": E_param.detach(),
                "V_param": V_param.detach(),
                "cohesion_param": cohesion_param.detach(),
                "friction_param": friction_param.detach(),
                "plastic_strain_param": plastic_strain_param.detach(),
            }
        )

        fu_pack = StokesSystem.build_sparse_Fu_by_coloring(
            u=u.detach(),
            mesh_state=mesh_state,
            density_x=density_x.detach(),
            density_y=density_y.detach(),
            tkp=tkp.detach(),
            gx=gx.detach(),
            gy=gy.detach(),
            dt=dt.detach(),
            pscale=pscale.detach(),
            boundary_const=boundary_const,
            timestep=timestep,
            viscosity_update_fn=viscosity_update_fn,
            rheology_context=rheology_context_for_Fu,
            p_fix_value=p_fix_value.detach(),
            is_stick_air=is_stick_air,
            coloring_pack=coloring_pack,
            create_sparse_tensor=False,
            return_csr=False,
            debug_print=False,
        )

        rows = fu_pack["rows"]
        cols = fu_pack["cols"]
        vals = fu_pack["vals"]
        N = int(fu_pack["N"])

        pattern = solvers.make_csr_pattern_from_coo(rows, cols, N)
        val_csr = solvers.coalesce_values_to_csr(vals, pattern)

        lambda_adj = solvers.csr_solve_from_values(
            crow=pattern.crow,
            col=pattern.col,
            row_of_val=pattern.row_of_val,
            val_csr=val_csr.detach(),
            b=grad_u.reshape(-1).detach(),
            shape=pattern.shape,
            matrix_key="stokes_fu_adjoint",
            reuse_mode="structure_fixed",
            transpose=True,
        )

        with torch.enable_grad():
            density_x_req = density_x.detach().clone().requires_grad_(True)
            density_y_req = density_y.detach().clone().requires_grad_(True)
            tkp_req = tkp.detach().clone().requires_grad_(True)

            A_req = A_param.detach().clone().requires_grad_(True)
            n_req = n_param.detach().clone().requires_grad_(True)
            E_req = E_param.detach().clone().requires_grad_(True)
            V_req = V_param.detach().clone().requires_grad_(True)

            cohesion_req = cohesion_param.detach().clone().requires_grad_(True)
            friction_req = friction_param.detach().clone().requires_grad_(True)
            plastic_strain_req = plastic_strain_param.detach().clone().requires_grad_(True)

            dt_req = dt.detach().clone().requires_grad_(True)
            pscale_req = pscale.detach().clone().requires_grad_(True)
            gx_req = gx.detach().clone().requires_grad_(True)
            gy_req = gy.detach().clone().requires_grad_(True)
            p_fix_req = p_fix_value.detach().clone().requires_grad_(True)

            rheology_context_req = dict(rheology_static_context)
            rheology_context_req.update(
                {
                    "A_param": A_req,
                    "n_param": n_req,
                    "E_param": E_req,
                    "V_param": V_req,
                    "cohesion_param": cohesion_req,
                    "friction_param": friction_req,
                    "plastic_strain_param": plastic_strain_req,
                }
            )

            pack = StokesSystem.build_stokes_residual_from_u(
                u=u.detach(),
                mesh_state=mesh_state,
                density_x=density_x_req,
                density_y=density_y_req,
                tkp=tkp_req,
                gx=gx_req,
                gy=gy_req,
                dt=dt_req,
                pscale=pscale_req,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_context=rheology_context_req,
                p_fix_value=p_fix_req,
                is_stick_air=is_stick_air,
                detach_density=False,
                return_sparse_tensor=False,
            )

            F = pack["F"]

            inputs = [
                density_x_req,
                density_y_req,
                tkp_req,
                A_req,
                n_req,
                E_req,
                V_req,
                cohesion_req,
                friction_req,
                plastic_strain_req,
                dt_req,
                pscale_req,
                gx_req,
                gy_req,
                p_fix_req,
            ]

            refs = [
                density_x,
                density_y,
                tkp,
                A_param,
                n_param,
                E_param,
                V_param,
                cohesion_param,
                friction_param,
                plastic_strain_param,
                dt,
                pscale,
                gx,
                gy,
                p_fix_value,
            ]

            grad_list = torch.autograd.grad(
                outputs=F,
                inputs=inputs,
                grad_outputs=lambda_adj,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        # Minus sign from implicit differentiation.
        grad_list = tuple(
            torch.zeros_like(ref) if g is None else -g
            for g, ref in zip(grad_list, refs)
        )

        return grad_list

    @staticmethod
    def _implicit_adjoint_eval_graph(
        *,
        u,
        density_x,
        density_y,
        tkp,
        A_param,
        n_param,
        E_param,
        V_param,
        cohesion_param,
        friction_param,
        plastic_strain_param,
        dt,
        pscale,
        gx,
        gy,
        p_fix_value,
        grad_u,
        mesh_state,
        boundary_const,
        timestep,
        viscosity_update_fn,
        rheology_static_context,
        is_stick_air,
        coloring_pack,
    ):
        if rheology_static_context is None:
            rheology_static_context = {}

        # Higher-order adjoint.
        rheology_context_for_Fu = dict(rheology_static_context)
        rheology_context_for_Fu.update(
            {
                "A_param": A_param,
                "n_param": n_param,
                "E_param": E_param,
                "V_param": V_param,
                "cohesion_param": cohesion_param,
                "friction_param": friction_param,
                "plastic_strain_param": plastic_strain_param,
            }
        )

        fu_pack = StokesSystem.build_sparse_Fu_by_coloring_graph(
            u=u,
            mesh_state=mesh_state,
            density_x=density_x,
            density_y=density_y,
            tkp=tkp,
            gx=gx,
            gy=gy,
            dt=dt,
            pscale=pscale,
            boundary_const=boundary_const,
            timestep=timestep,
            viscosity_update_fn=viscosity_update_fn,
            rheology_context=rheology_context_for_Fu,
            p_fix_value=p_fix_value,
            is_stick_air=is_stick_air,
            coloring_pack=coloring_pack,
            chunk_size=None,
            debug_print=False,
        )

        rows = fu_pack["rows"]
        cols = fu_pack["cols"]
        vals = fu_pack["vals"]
        N = int(fu_pack["N"])

        pattern = solvers.make_csr_pattern_from_coo(rows, cols, N)
        val_csr = solvers.coalesce_values_to_csr(vals, pattern)

        lambda_adj = solvers.csr_solve_from_values(
            crow=pattern.crow,
            col=pattern.col,
            row_of_val=pattern.row_of_val,
            val_csr=val_csr,
            b=grad_u.reshape(-1),
            shape=pattern.shape,
            matrix_key="stokes_fu_adjoint_graph",
            reuse_mode="structure_fixed",
            transpose=True,
        )

        rheology_context_req = dict(rheology_static_context)
        rheology_context_req.update(
            {
                "A_param": A_param,
                "n_param": n_param,
                "E_param": E_param,
                "V_param": V_param,
                "cohesion_param": cohesion_param,
                "friction_param": friction_param,
                "plastic_strain_param": plastic_strain_param,
            }
        )

        pack = StokesSystem.build_stokes_residual_from_u(
            u=u,
            mesh_state=mesh_state,
            density_x=density_x,
            density_y=density_y,
            tkp=tkp,
            gx=gx,
            gy=gy,
            dt=dt,
            pscale=pscale,
            boundary_const=boundary_const,
            timestep=timestep,
            viscosity_update_fn=viscosity_update_fn,
            rheology_context=rheology_context_req,
            p_fix_value=p_fix_value,
            is_stick_air=is_stick_air,
            detach_density=False,
            return_sparse_tensor=False,
        )

        F = pack["F"]

        inputs = [
            density_x,
            density_y,
            tkp,
            A_param,
            n_param,
            E_param,
            V_param,
            cohesion_param,
            friction_param,
            plastic_strain_param,
            dt,
            pscale,
            gx,
            gy,
            p_fix_value,
        ]

        grad_list = torch.autograd.grad(
            outputs=F,
            inputs=inputs,
            grad_outputs=lambda_adj,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )

        refs = inputs

        # Preserve the implicit-differentiation sign and tuple structure.
        grad_list = tuple(
            torch.zeros_like(ref) if g is None else -g
            for g, ref in zip(grad_list, refs)
        )

        return grad_list

    @staticmethod
    def forward(
        ctx,
        density_x,
        density_y,
        tkp,
        A_param,
        n_param,
        E_param,
        V_param,
        cohesion_param,
        friction_param,
        plastic_strain_param,
        dt,
        pscale,
        gx,
        gy,
        p_fix_value,
        mesh_state,
        boundary_const,
        timestep,
        viscosity_update_fn,
        rheology_static_context,
        is_stick_air,
        u0,
        coloring_pack,
        picard_steps,
        newton_steps,
        rel_tol,
        line_search,
    ):
        """
        Forward does nonlinear solve, no unroll graph is kept.
        """

        if rheology_static_context is None:
            rheology_static_context = {}

        rheology_context_forward = dict(rheology_static_context)
        rheology_context_forward.update(
            {
                "A_param": A_param,
                "n_param": n_param,
                "E_param": E_param,
                "V_param": V_param,
                "cohesion_param": cohesion_param,
                "friction_param": friction_param,
                "plastic_strain_param": plastic_strain_param,
            }
        )
        with torch.no_grad():
            state = StokesSystem.nonlinear_solve_fu(
                mesh_state=mesh_state,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                gx=gx,
                gy=gy,
                dt=dt,
                pscale=pscale,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_context=rheology_context_forward,
                p_fix_value=p_fix_value,
                is_stick_air=is_stick_air,
                u0=u0,
                picard_steps=picard_steps,
                newton_steps=newton_steps,
                rel_tol=rel_tol,
                coloring_pack=coloring_pack,
                line_search = line_search,
            )

            u = state["u"]

        ctx.mesh_state = mesh_state
        ctx.boundary_const = boundary_const
        ctx.timestep = timestep
        ctx.viscosity_update_fn = viscosity_update_fn
        ctx.rheology_static_context = rheology_static_context
        ctx.is_stick_air = is_stick_air
        ctx.coloring_pack = state.get("coloring_pack", coloring_pack)

        ctx.save_for_backward(
            u,
            density_x,
            density_y,
            tkp,
            A_param,
            n_param,
            E_param,
            V_param,
            cohesion_param,
            friction_param,
            plastic_strain_param,
            dt,
            pscale,
            gx,
            gy,
            p_fix_value,
        )

        return u

    @staticmethod
    def backward(ctx, grad_u):
        (
            u,
            density_x,
            density_y,
            tkp,
            A_param,
            n_param,
            E_param,
            V_param,
            cohesion_param,
            friction_param,
            plastic_strain_param,
            dt,
            pscale,
            gx,
            gy,
            p_fix_value,
        ) = ctx.saved_tensors

        mesh_state = ctx.mesh_state
        boundary_const = ctx.boundary_const
        timestep = ctx.timestep
        viscosity_update_fn = ctx.viscosity_update_fn
        rheology_static_context = ctx.rheology_static_context
        is_stick_air = ctx.is_stick_air
        coloring_pack = ctx.coloring_pack

        build_graph = torch.is_grad_enabled()

        if build_graph:
            grad_tuple = StokesImplicitAdjointFn.apply(
                u,
                density_x,
                density_y,
                tkp,
                A_param,
                n_param,
                E_param,
                V_param,
                cohesion_param,
                friction_param,
                plastic_strain_param,
                dt,
                pscale,
                gx,
                gy,
                p_fix_value,
                grad_u,
                mesh_state,
                boundary_const,
                timestep,
                viscosity_update_fn,
                rheology_static_context,
                is_stick_air,
                coloring_pack,
            )
        else:
            grad_tuple = StokesImplicitSolveFn._implicit_adjoint_eval_numeric(
                u=u,
                density_x=density_x,
                density_y=density_y,
                tkp=tkp,
                A_param=A_param,
                n_param=n_param,
                E_param=E_param,
                V_param=V_param,
                cohesion_param=cohesion_param,
                friction_param=friction_param,
                plastic_strain_param=plastic_strain_param,
                dt=dt,
                pscale=pscale,
                gx=gx,
                gy=gy,
                p_fix_value=p_fix_value,
                grad_u=grad_u,
                mesh_state=mesh_state,
                boundary_const=boundary_const,
                timestep=timestep,
                viscosity_update_fn=viscosity_update_fn,
                rheology_static_context=rheology_static_context,
                is_stick_air=is_stick_air,
                coloring_pack=coloring_pack,
            )

        (
            grad_density_x,
            grad_density_y,
            grad_tkp,
            grad_A,
            grad_n,
            grad_E,
            grad_V,
            grad_cohesion,
            grad_friction,
            grad_plastic_strain,
            grad_dt,
            grad_pscale,
            grad_gx,
            grad_gy,
            grad_p_fix,
        ) = grad_tuple

        needs = ctx.needs_input_grad

        def maybe_grad(index, grad):
            return grad if needs[index] else None

        return (
            maybe_grad(0, grad_density_x),
            maybe_grad(1, grad_density_y),
            maybe_grad(2, grad_tkp),
            maybe_grad(3, grad_A),
            maybe_grad(4, grad_n),
            maybe_grad(5, grad_E),
            maybe_grad(6, grad_V),
            maybe_grad(7, grad_cohesion),
            maybe_grad(8, grad_friction),
            maybe_grad(9, grad_plastic_strain),
            maybe_grad(10, grad_dt),
            maybe_grad(11, grad_pscale),
            maybe_grad(12, grad_gx),
            maybe_grad(13, grad_gy),
            maybe_grad(14, grad_p_fix),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # picard_steps
            None,  # newton_steps
            None,  # rel_tol
            None,  # line_search
        )

def stokes_implicit_solve(
    mesh_state,
    density_x,
    density_y,
    tkp,
    gx,
    gy,
    dt,
    pscale,
    boundary_const,
    timestep,
    *,
    viscosity_update_fn,
    rheology_context=None,
    p_fix_value=None,
    is_stick_air=False,
    u0=None,
    coloring_pack=None,
    picard_steps=None,
    newton_steps=None,
    rel_tol=None,
    line_search=True,
):
    """
    User-facing wrapper.

    It unpacks rheology_context into:
        differentiable Tensor inputs:
            A_param, n_param, E_param, V_param

        static context:
            weakzone_mask_node, weakzone_mask_p, epsII_ref_nd, etc.

    Then forward/backward re-pack rheology_context internally.
    """

    if rheology_context is None:
        rheology_context = {}

    A_param = rheology_context.get("A_param", None)
    n_param = rheology_context.get("n_param", None)
    E_param = rheology_context.get("E_param", None)
    V_param = rheology_context.get("V_param", None)
    cohesion_param = rheology_context.get("cohesion_param", None)
    friction_param = rheology_context.get("friction_param", None)
    plastic_strain_param = rheology_context.get("plastic_strain_param", None)

    def _dummy_like_tkp():
        return torch.zeros((), dtype=tkp.dtype, device=tkp.device)

    if A_param is None:
        A_param = _dummy_like_tkp()
    if n_param is None:
        n_param = _dummy_like_tkp()
    if E_param is None:
        E_param = _dummy_like_tkp()
    if V_param is None:
        V_param = _dummy_like_tkp()
    if cohesion_param is None:
        cohesion_param = _dummy_like_tkp()
    if friction_param is None:
        friction_param = _dummy_like_tkp()
    if plastic_strain_param is None:
        plastic_strain_param = _dummy_like_tkp()

    rheology_static_context = dict(rheology_context)
    rheology_static_context.pop("A_param", None)
    rheology_static_context.pop("n_param", None)
    rheology_static_context.pop("E_param", None)
    rheology_static_context.pop("V_param", None)
    rheology_static_context.pop("cohesion_param", None)
    rheology_static_context.pop("friction_param", None)
    rheology_static_context.pop("plastic_strain_param", None)

    if p_fix_value is None:
        p_fix_value = torch.tensor(
            0.0,
            dtype=tkp.dtype,
            device=tkp.device,
        )

    if not torch.is_tensor(gx):
        gx = torch.tensor(
            gx,
            dtype=tkp.dtype,
            device=tkp.device,
        )

    if not torch.is_tensor(gy):
        gy = torch.tensor(
            gy,
            dtype=tkp.dtype,
            device=tkp.device,
        )

    if not torch.is_tensor(dt):
        dt = torch.tensor(
            dt,
            dtype=tkp.dtype,
            device=tkp.device,
        )

    if not torch.is_tensor(pscale):
        pscale = torch.tensor(
            pscale,
            dtype=tkp.dtype,
            device=tkp.device,
        )


    return StokesImplicitSolveFn.apply(
        density_x,
        density_y,
        tkp,
        A_param,
        n_param,
        E_param,
        V_param,
        cohesion_param,
        friction_param,
        plastic_strain_param,
        dt,
        pscale,
        gx,
        gy,
        p_fix_value,
        mesh_state,
        boundary_const,
        timestep,
        viscosity_update_fn,
        rheology_static_context,
        is_stick_air,
        u0,
        coloring_pack,
        picard_steps,
        newton_steps,
        rel_tol,
        line_search,
    )
