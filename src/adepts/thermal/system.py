import torch
from .assembly import thermal_assembly
from .. import solvers
from ..advection import advect_temperature


class ThermalSystemGrid:
    @staticmethod
    def T_extract(S: torch.Tensor, mesh_state):
        Nx1 = mesh_state["Nx1"]
        Ny1 = mesh_state["Ny1"]
        assert S.numel() == Nx1 * Ny1
        return S.view(Nx1, Ny1).transpose(0, 1).contiguous()

    @staticmethod
    def apply_temperature_bc_on_p(T: torch.Tensor, thermal_boundary_dict, mesh_state):
        """
        Apply temperature ghost-cell boundary conditions on the full p-grid.

        Dirichlet:
            T_ghost = 2*T_bc - T_interior

        Neumann:
            value is the outward-normal derivative dT/dn.

            top    : n = -y
            bottom : n = +y
            left   : n = -x
            right  : n = +x
        """
        if thermal_boundary_dict is None:
            return T

        T = T.clone()

        if T.ndim != 2:
            raise ValueError(f"T must be 2D, got shape={tuple(T.shape)}")

        dtype = T.dtype
        device = T.device

        xp = mesh_state["xp"].to(device=device, dtype=dtype)
        yp = mesh_state["yp"].to(device=device, dtype=dtype)

        expected_shape = (yp.numel(), xp.numel())
        if T.shape != expected_shape:
            raise ValueError(
                f"T shape mismatch: got {tuple(T.shape)}, expected {expected_shape}"
            )

        dx_left = xp[1] - xp[0]
        dx_right = xp[-1] - xp[-2]
        dy_top = yp[1] - yp[0]
        dy_bottom = yp[-1] - yp[-2]

        def _to_tensor_value(v):
            if torch.is_tensor(v):
                return v.to(device=device, dtype=dtype)
            return torch.tensor(v, device=device, dtype=dtype)

        def get_bc(side):
            cfg = thermal_boundary_dict.get(side, None)
            if cfg is None:
                return None, None
            return cfg["type"].lower(), _to_tensor_value(cfg.get("value", 0.0))

        btype, bval = get_bc("top")
        if btype == "dirichlet":
            T[0, 1:-1] = 2.0 * bval - T[1, 1:-1]
        elif btype == "neumann":
            T[0, 1:-1] = T[1, 1:-1] + bval * dy_top
        elif btype is not None:
            raise ValueError(f"Unknown BC type on top: {btype}")

        btype, bval = get_bc("bottom")
        if btype == "dirichlet":
            T[-1, 1:-1] = 2.0 * bval - T[-2, 1:-1]
        elif btype == "neumann":
            T[-1, 1:-1] = T[-2, 1:-1] + bval * dy_bottom
        elif btype is not None:
            raise ValueError(f"Unknown BC type on bottom: {btype}")

        btype, bval = get_bc("left")
        if btype == "dirichlet":
            T[1:-1, 0] = 2.0 * bval - T[1:-1, 1]
        elif btype == "neumann":
            T[1:-1, 0] = T[1:-1, 1] + bval * dx_left
        elif btype is not None:
            raise ValueError(f"Unknown BC type on left: {btype}")

        btype, bval = get_bc("right")
        if btype == "dirichlet":
            T[1:-1, -1] = 2.0 * bval - T[1:-1, -2]
        elif btype == "neumann":
            T[1:-1, -1] = T[1:-1, -2] + bval * dx_right
        elif btype is not None:
            raise ValueError(f"Unknown BC type on right: {btype}")

        T[0, 0] = 0.5 * (T[0, 1] + T[1, 0])
        T[0, -1] = 0.5 * (T[0, -2] + T[1, -1])
        T[-1, 0] = 0.5 * (T[-1, 1] + T[-2, 0])
        T[-1, -1] = 0.5 * (T[-1, -2] + T[-2, -1])

        return T


    @staticmethod
    def diffuse_temperature_implicit(
        T_in: torch.Tensor,
        KX: torch.Tensor,
        KY: torch.Tensor,
        RHOCP: torch.Tensor,
        dt: torch.Tensor,
        HSUM: torch.Tensor,
        mesh_state,
        timestep: int,
        thermal_boundary_dict,
    ):
        T_in = ThermalSystemGrid.apply_temperature_bc_on_p(
            T_in, thermal_boundary_dict,mesh_state
        )

        rows, cols, vals, R = thermal_assembly(
            KX=KX,
            KY=KY,
            RHOCP=RHOCP,
            dt=dt,
            tk1=T_in,
            HSUM=HSUM,
            mesh_state=mesh_state,
            thermal_boundary_dict=thermal_boundary_dict,
        )

        Nx1 = int(mesh_state["Nx1"])
        Ny1 = int(mesh_state["Ny1"])
        Nnode = Nx1 * Ny1

        pattern = solvers.make_csr_pattern_from_coo(
            rows,
            cols,
            Nnode,
        )

        val_csr = solvers.coalesce_values_to_csr(
            vals,
            pattern,
        )

        T_out_flat = solvers.csr_solve_from_values(
            crow=pattern.crow,
            col=pattern.col,
            row_of_val=pattern.row_of_val,
            val_csr=val_csr,
            b=R,
            shape=pattern.shape,
            matrix_key="thermal",
            reuse_mode="structure_fixed",
            transpose=False,
        )

        T_out = ThermalSystemGrid.T_extract(T_out_flat, mesh_state)

        T_out = ThermalSystemGrid.apply_temperature_bc_on_p(
            T_out,
            thermal_boundary_dict,
            mesh_state,
        )

        return T_out

    @staticmethod
    def advance_temperature_strang(
        T_old: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        KX: torch.Tensor,
        KY: torch.Tensor,
        RHOCP: torch.Tensor,
        dt: torch.Tensor,
        mesh_state,
        timestep: int,
        thermal_boundary_dict,
        HSUM: torch.Tensor = None,
        *,
        split_order: str = "DAD",
        edge_mode: str = "nearest",
        constant_value: float = 0.0,
        backtrace: str = "rk2",
        recompute_coeff_second_half: bool = False,
        coeff_update_fn=None,
        coeff_context=None,
    ):
        """Advance one DAD or ADA Strang-split time step."""
        split_order = split_order.upper()
        if split_order not in ("DAD", "ADA"):
            raise ValueError(f"split_order must be 'DAD' or 'ADA', got {split_order!r}")

        if HSUM is None:
            HSUM = torch.zeros_like(T_old)

        T0 = ThermalSystemGrid.apply_temperature_bc_on_p(
            T_old, thermal_boundary_dict,mesh_state
        )


        T_bg_adv_use = None

        def _advect_temperature(T_in: torch.Tensor, adv_dt: torch.Tensor) -> torch.Tensor:
            """
            semi-Lagrangian advection.
            """

            q_old = T_in
            q_adv_full = advect_temperature(
                q_old=q_old,
                vx=vx,
                vy=vy,
                dt=adv_dt,
                mesh_state=mesh_state,
                edge_mode=edge_mode,
                constant_value=constant_value,
                scalar_boundary_dict=thermal_boundary_dict,
                backtrace=backtrace,
                enforce_bc_after_advect=True,
                use_dirichlet_value_for_oob=True,
            )
            T_adv_full = q_adv_full

            T_out = T_in.clone()
            T_out[1:-1, 1:-1] = T_adv_full[1:-1, 1:-1]

            T_out = ThermalSystemGrid.apply_temperature_bc_on_p(
                T_out, thermal_boundary_dict,mesh_state
            )
            return T_out

        def _diffuse_temperature(
            T_in: torch.Tensor,
            diff_dt: torch.Tensor,
            KX_in: torch.Tensor,
            KY_in: torch.Tensor,
            RHOCP_in: torch.Tensor,
        ):
            return ThermalSystemGrid.diffuse_temperature_implicit(
                T_in=T_in,
                KX=KX_in,
                KY=KY_in,
                RHOCP=RHOCP_in,
                dt=diff_dt,
                HSUM=HSUM,
                mesh_state=mesh_state,
                timestep=timestep,
                thermal_boundary_dict=thermal_boundary_dict,
            )

        if split_order == "DAD":
            T_stage1 = _diffuse_temperature(
                T_in=T0,
                diff_dt=0.5 * dt,
                KX_in=KX,
                KY_in=KY,
                RHOCP_in=RHOCP,
            )

            T_stage2 = _advect_temperature(T_stage1, dt)

            if recompute_coeff_second_half:
                if coeff_update_fn is None:
                    raise ValueError(
                        "recompute_coeff_second_half=True but coeff_update_fn is None"
                    )
                KX2, KY2, RHOCP2 = coeff_update_fn(T_stage2, coeff_context)
            else:
                KX2, KY2, RHOCP2 = KX, KY, RHOCP

            T_new = _diffuse_temperature(
                T_in=T_stage2,
                diff_dt=0.5 * dt,
                KX_in=KX2,
                KY_in=KY2,
                RHOCP_in=RHOCP2,
            )

            return {"T": T_new}

        elif split_order == "ADA":
            T_stage1 = _advect_temperature(T0, 0.5 * dt)

            if recompute_coeff_second_half:
                if coeff_update_fn is None:
                    raise ValueError(
                        "recompute_coeff_second_half=True but coeff_update_fn is None"
                    )
                KX_mid, KY_mid, RHOCP_mid = coeff_update_fn(T_stage1, coeff_context)
            else:
                KX_mid, KY_mid, RHOCP_mid = KX, KY, RHOCP

            T_stage2 = _diffuse_temperature(
                T_in=T_stage1,
                diff_dt=dt,
                KX_in=KX_mid,
                KY_in=KY_mid,
                RHOCP_in=RHOCP_mid,
            )

            T_new = _advect_temperature(T_stage2, 0.5 * dt)

            return {"T": T_new}
