import torch
from math import ceil

class CartesianMesh:
    def __init__(self,
                 xsize, ysize,
                 xbase_resolution, ybase_resolution,
                 xfine_ranges=None, xfine_resolution=None, xtransition_distance=None,
                 yfine_ranges=None, yfine_resolution=None, ytransition_distance=None,
                 device="default"):
        self._frozen = False
        if device == "default":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available.")
            self.device = torch.device("cuda")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device type: {device}")

        device = self.device
        dtype = torch.float64

        def to_tensor(x):
            device = self.device
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype, device=device)
            elif isinstance(x, (float, int)):
                return torch.tensor(x, dtype=dtype, device=device)
            elif hasattr(x, 'shape'):
                return torch.tensor(x, dtype=dtype, device=device)
            else:
                raise TypeError(f"Unsupported input type: {type(x)}")

        xsize = to_tensor(xsize)
        ysize = to_tensor(ysize)
        xbase_resolution = to_tensor(xbase_resolution)
        ybase_resolution = to_tensor(ybase_resolution)

        if xfine_ranges is None or xfine_resolution is None or xtransition_distance is None:
            ncell_x = int((xsize / xbase_resolution).item())
            self.xnode = torch.linspace(0, xsize.item(), steps=ncell_x + 1, dtype=dtype, device=device)
        else:
            self.xnode = self.generate_custom_array(
                xsize.item(), xbase_resolution.item(), xfine_ranges,
                xfine_resolution, xtransition_distance,
                device=device)

        if yfine_ranges is None or yfine_resolution is None or ytransition_distance is None:
            ncell_y = int((ysize / ybase_resolution).item())
            self.ynode = torch.linspace(0, ysize.item(), steps=ncell_y + 1, dtype=dtype, device=device)
        else:
            self.ynode = self.generate_custom_array(
                ysize.item(), ybase_resolution.item(), yfine_ranges,
                yfine_resolution, ytransition_distance,
                device=device)

        self.xsize = xsize
        self.ysize = ysize
        self.Nx = self.xnode.size(0)
        self.Ny = self.ynode.size(0)
        self.Nx1 = self.Nx + 1
        self.Ny1 = self.Ny + 1
        self.dx = self.xsize / (self.Nx - 1)
        self.dy = self.ysize / (self.Ny - 1)
        self.shape = [self.Ny,self.Nx]
        dx_right = self.xnode[-1] - self.xnode[-2]
        self.xvx = torch.cat([self.xnode, self.xnode[-1:] + dx_right])
        self.yvx = torch.zeros(self.Ny + 1, dtype=dtype, device=device)
        self.yvx[1:self.Ny] = 0.5 * (self.ynode[:-1] + self.ynode[1:])
        self.yvx[0] = self.yvx[1] - 2.0 * (self.yvx[1] - self.ynode[0])
        self.yvx[-1] = self.yvx[-2] + 2.0 * (self.ynode[-1] - self.yvx[-2])

        dy_top = self.ynode[-1] - self.ynode[-2]
        self.yvy = torch.cat([self.ynode, self.ynode[-1:] + dy_top])
        self.xvy = torch.zeros(self.Nx + 1, dtype=dtype, device=device)
        self.xvy[1:self.Nx] = 0.5 * (self.xnode[:-1] + self.xnode[1:])
        self.xvy[0] = self.xvy[1] - 2.0 * (self.xvy[1] - self.xnode[0])
        self.xvy[-1] = self.xvy[-2] + 2.0 * (self.xnode[-1] - self.xvy[-2])

        self.xp = self.xvy.clone()
        self.yp = self.yvx.clone()

        self.point_array = self.generate_point_index_array()


        self.freeze()
    def freeze(self):
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and hasattr(self, name):
            raise AttributeError(f"Cannot modify frozen attribute '{name}'")
        super().__setattr__(name, value)

    def __repr__(self):
        grid_info = (
            f"<CartesianMesh: Nx={self.Nx}, Ny={self.Ny}\n"
            f"  Domain Size: xsize={self.xsize:.1e} m, ysize={self.ysize:.1e} m\n"
            f"  Grid Types Available: ['xnode', 'ynode', 'xvx', 'yvx', 'xvy', 'yvy', 'xp', 'yp', 'point_array']\n"
            f">"
        )
        return grid_info

    @staticmethod
    def preprocess_fine_ranges(fine_ranges: list[tuple[float, float]],
                               transition_distance: float) -> list[tuple[float, float]]:
        """Merge refined regions separated by less than two transitions."""
        if not fine_ranges:
            return []

        fine_ranges = sorted(fine_ranges, key=lambda x: (x[0], x[1]))

        merged = []
        cur_s, cur_e = fine_ranges[0]

        for i in range(1, len(fine_ranges)):
            nxt_s, nxt_e = fine_ranges[i]
            if (nxt_s - cur_e) < 2.0 * transition_distance:
                cur_e = max(cur_e, nxt_e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = nxt_s, nxt_e

        merged.append((cur_s, cur_e))
        return merged

    @staticmethod
    def generate_custom_array(base_range: torch.Tensor | float,
                              base_resolution: torch.Tensor | float,
                              fine_ranges: list[tuple[float, float]] | None,
                              fine_resolution: torch.Tensor | float | None,
                              transition_distance: torch.Tensor | float | None,
                              *,
                              device="cpu",
                              dtype=torch.float64):


        def to_float(x):
            return x.item() if isinstance(x, torch.Tensor) else float(x)

        def empty():
            return torch.tensor([], device=device, dtype=dtype)

        def arange_pos_safe(start: float, end: float, step: float):
            if step <= 0:
                raise ValueError("step must be positive.")
            if end <= start:
                return empty()
            return torch.arange(start, end, step=step, device=device, dtype=dtype)

        base_range      = to_float(base_range)
        base_resolution = to_float(base_resolution)

        if fine_ranges is None or fine_resolution is None or transition_distance is None:
            if base_resolution <= 0:
                raise ValueError("base_resolution must be positive.")
            n_cell = int(base_range / base_resolution)
            return torch.linspace(0.0, base_range,
                                  steps=n_cell + 1,
                                  dtype=dtype,
                                  device=device)

        fine_resolution    = to_float(fine_resolution)
        transition_distance = to_float(transition_distance)
        if base_resolution <= 0 or fine_resolution <= 0:
            raise ValueError("base_resolution and fine_resolution must be positive.")

        fine_ranges = CartesianMesh.preprocess_fine_ranges(fine_ranges, transition_distance)

        base_list = []
        cur_start = 0.0
        for (f_s, f_e) in fine_ranges:
            seg_end = f_s - transition_distance
            base_list.append(arange_pos_safe(cur_start, seg_end, base_resolution))
            cur_start = f_e + transition_distance
        base_list.append(arange_pos_safe(cur_start, base_range + base_resolution, base_resolution))
        base_array = torch.cat([t for t in base_list if t.numel() > 0]) if base_list else empty()

        fine_arrays = []
        for (f_s, f_e) in fine_ranges:
            length = max(0.0, f_e - f_s)
            n_step = int(ceil(length / fine_resolution))
            f_end_adj = f_s + n_step * fine_resolution
            arr = arange_pos_safe(f_s, f_end_adj + fine_resolution * 0.5, fine_resolution)
            fine_arrays.append(arr)

        trans_arrays = []
        avg_dx = 0.5 * (base_resolution + fine_resolution)
        avg_dx = avg_dx if avg_dx > 0 else min(base_resolution, fine_resolution)

        n_trans = max(1, int(transition_distance // avg_dx))
        for f_arr in fine_arrays:
            if f_arr.numel() == 0:
                continue
            t_front = torch.linspace(float(f_arr[0]) - transition_distance,
                                     float(f_arr[0]),
                                     steps=n_trans + 1,
                                     device=device,
                                     dtype=dtype)[:-1]
            t_back  = torch.linspace(float(f_arr[-1]),
                                     float(f_arr[-1]) + transition_distance,
                                     steps=n_trans + 1,
                                     device=device,
                                     dtype=dtype)[1:]
            trans_arrays.extend([t_front, t_back])

        pieces = [base_array, *fine_arrays, *trans_arrays]
        pieces = [p for p in pieces if p.numel() > 0]
        combined = torch.cat(pieces) if pieces else empty()

        if combined.numel() == 0:
            return torch.tensor([0.0, base_range], device=device, dtype=dtype)

        combined = torch.unique(combined)
        combined = combined[(combined >= 0.0) & (combined <= base_range)]
        if combined.numel() == 0:
            combined = torch.tensor([0.0, base_range], device=device, dtype=dtype)
        else:
            combined[0]  = 0.0
            combined[-1] = base_range

        return combined

    def generate_point_index_array(self):
        """Map grid points to column-major scalar and staggered DOFs."""

        Ny1, Nx1 = self.Ny1, self.Nx1
        device   = self.xnode.device
        kv = torch.zeros((Ny1, Nx1, 4), dtype=torch.int64, device=device)

        ii = torch.arange(Ny1, dtype=torch.int64, device=device).unsqueeze(1).repeat(1, Nx1)
        jj = torch.arange(Nx1, dtype=torch.int64, device=device).unsqueeze(0).repeat(Ny1, 1)
        kv[:, :, 0] = (jj * Ny1 + ii)

        kvx = (jj * Ny1 + ii) * 3
        kv[:, :, 1] = kvx
        kv[:, :, 2] = kvx + 1
        kv[:, :, 3] = kvx + 2
        return kv

    def to_mesh_state(self, *, detach: bool = True, cpu: bool = False) -> dict:
        def pack(t: torch.Tensor):
            if detach:
                t = t.detach()
            if cpu:
                t = t.to("cpu")
            return t.clone()

        state = {
            "Nx": int(self.Nx),
            "Ny": int(self.Ny),
            "Nx1": int(self.Nx1),
            "Ny1": int(self.Ny1),

            "xsize": float(self.xsize.item()) if isinstance(self.xsize, torch.Tensor) else float(self.xsize),
            "ysize": float(self.ysize.item()) if isinstance(self.ysize, torch.Tensor) else float(self.ysize),

            "xnode": pack(self.xnode),
            "ynode": pack(self.ynode),
            "xvx":   pack(self.xvx),
            "yvx":   pack(self.yvx),
            "xvy":   pack(self.xvy),
            "yvy":   pack(self.yvy),
            "xp":    pack(self.xp),
            "yp":    pack(self.yp),


            "point_array": pack(self.point_array),

            "dtype": str(self.xnode.dtype),
            "device": str(self.xnode.device),
        }
        return state



def get_nondimensional_mesh_state(mesh_state: dict, L0: float) -> dict:
    nd_state = mesh_state.copy()

    spatial_keys = [
        "xsize", "ysize",
        "xnode", "ynode",
        "xvx", "yvx",
        "xvy", "yvy",
        "xp", "yp"
    ]

    for key in spatial_keys:
        if key in nd_state:
            val = nd_state[key]
            if isinstance(val, torch.Tensor):
                nd_state[key] = val / L0
            elif isinstance(val, (float, int)):
                nd_state[key] = float(val) / L0

    return nd_state
