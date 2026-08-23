"""
engine/python_bindings.py

Python bindings for the C++ SIMD Gridiron Engine.
Uses ctypes to load the shared library and provides a clean Python interface
for the `ml/season_simulator.py` to use.
"""

import ctypes
import platform
from pathlib import Path

import numpy as np

# ── Dynamic Library Loader ────────────────────────────────────────────────────

def _load_engine_lib():
    """Finds and loads the compiled C++ shared library."""
    system = platform.system()
    ext = ".dylib" if system == "Darwin" else ".so"
    
    # Try different build paths
    base_dir = Path(__file__).resolve().parent
    paths = [
        base_dir / "build" / f"libgridiron_engine{ext}",
        base_dir / "build" / "Release" / f"libgridiron_engine{ext}",
        base_dir / "build" / "Debug" / f"libgridiron_engine{ext}"
    ]
    
    for path in paths:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                return lib
            except BaseException:
                pass
                
    # If not found, return None (caller should fallback to Python simulator)
    return None


LIB = _load_engine_lib()

# ── Numpy DTypes for Structs ──────────────────────────────────────────────────

# Matches gridiron::PlayerProjection in C++ (56 bytes total)
PLAYER_DTYPE = np.dtype([
    ("player_id", "S32"),
    ("position", "S4"),
    ("team", "S4"),
    ("kalman_mean", "f4"),
    ("kalman_variance", "f4"),
    ("injury_multiplier", "f4"),
    ("snap_share", "f4"),
])

# ── C-API Signatures ──────────────────────────────────────────────────────────

if LIB is not None:
    # void* season_sim_create(uint32_t n_simulations, uint32_t rng_seed);
    LIB.season_sim_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    LIB.season_sim_create.restype = ctypes.c_void_p

    # void season_sim_destroy(void* handle);
    LIB.season_sim_destroy.argtypes = [ctypes.c_void_p]
    LIB.season_sim_destroy.restype = None

    # void season_sim_week(void* handle, const float* player_data, uint32_t n_players,
    #                      const float* corr_mat, float* out);
    LIB.season_sim_week.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=PLAYER_DTYPE, flags="C_CONTIGUOUS"),
        ctypes.c_uint32,
        ctypes.c_void_p,  # corr_mat (can be NULL)
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS, WRITEABLE")
    ]
    LIB.season_sim_week.restype = None

    # void season_sim_full(void* handle, const float* player_data, uint32_t n_players,
    #                      const float* corr_mat, uint32_t start_week,
    #                      uint32_t end_week, float* out_means, float* out_p10,
    #                      float* out_p50, float* out_p90);
    LIB.season_sim_full.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=PLAYER_DTYPE, flags="C_CONTIGUOUS"),
        ctypes.c_uint32,
        ctypes.c_void_p,  # corr_mat (can be NULL)
        ctypes.c_uint32,
        ctypes.c_uint32,
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS, WRITEABLE"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS, WRITEABLE"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS, WRITEABLE"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS, WRITEABLE")
    ]
    LIB.season_sim_full.restype = None

    LIB.drive_mcmc_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    LIB.drive_mcmc_create.restype = ctypes.c_void_p
    LIB.drive_mcmc_destroy.argtypes = [ctypes.c_void_p]
    LIB.drive_mcmc_destroy.restype = None
    LIB.drive_mcmc_load_defaults.argtypes = [ctypes.c_void_p]
    LIB.drive_mcmc_load_defaults.restype = None
    LIB.drive_mcmc_load_transitions_csv.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    LIB.drive_mcmc_load_transitions_csv.restype = ctypes.c_bool
    LIB.drive_mcmc_simulate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_int8,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    LIB.drive_mcmc_simulate.restype = None


# ── Python Wrapper Class ──────────────────────────────────────────────────────

class CppSeasonSimulator:
    """Python interface to the C++ gridiron::SeasonSimulator."""

    def __init__(self, n_simulations: int = 10000, rng_seed: int = 42):
        if LIB is None:
            raise RuntimeError("C++ engine shared library not found. Build it first.")
        self.handle = LIB.season_sim_create(n_simulations, rng_seed)
        self.n_simulations = n_simulations

    def __del__(self):
        if hasattr(self, "handle") and self.handle and LIB is not None:
            LIB.season_sim_destroy(self.handle)
            self.handle = None

    def sim_full_season(
        self,
        players_arr: np.ndarray,      # structured array of PLAYER_DTYPE
        corr_mat: np.ndarray,         # [n_players, n_players] float32 array
        start_week: int,
        end_week: int,
    ) -> dict[str, np.ndarray]:
        """
        Runs the full season simulation entirely in C++.
        
        Args:
            players_arr: 1D array of dtype PLAYER_DTYPE.
            corr_mat: 2D array of correlation data. Can be empty (None-equivalent) if passing zeros.
            start_week: Starting week.
            end_week: Ending week (inclusive).
            
        Returns:
            Dictionary with arrays maped by order corresponding to players_arr:
            {
              "mean": np.ndarray,
              "p10": np.ndarray,
              "p50": np.ndarray,
              "p90": np.ndarray
            }
        """
        n_players = len(players_arr)
        
        # Ensure contiguous data
        p_data = np.ascontiguousarray(players_arr, dtype=PLAYER_DTYPE)
        
        c_ptr = None
        if corr_mat is not None and corr_mat.size > 0:
            c_data = np.ascontiguousarray(corr_mat, dtype=np.float32)
            c_ptr = c_data.ctypes.data_as(ctypes.c_void_p)

        out_means = np.zeros(n_players, dtype=np.float32)
        out_p10 = np.zeros(n_players, dtype=np.float32)
        out_p50 = np.zeros(n_players, dtype=np.float32)
        out_p90 = np.zeros(n_players, dtype=np.float32)
        
        LIB.season_sim_full(
            self.handle,
            p_data,
            n_players,
            c_ptr,
            start_week,
            end_week,
            out_means, out_p10, out_p50, out_p90
        )
        
        return {
            "mean": out_means,
            "p10": out_p10,
            "p50": out_p50,
            "p90": out_p90
        }


class CppDriveMCMC:
    """Python interface to C++ DriveMCMC. Requires fitted transitions.csv."""

    def __init__(self, n_simulations: int = 2000, rng_seed: int = 42):
        if LIB is None:
            raise RuntimeError("C++ engine shared library not found. Build it first.")
        self.handle = LIB.drive_mcmc_create(n_simulations, rng_seed)
        self.n_simulations = n_simulations

    def __del__(self):
        if hasattr(self, "handle") and self.handle and LIB is not None:
            LIB.drive_mcmc_destroy(self.handle)
            self.handle = None

    def load_transitions_csv(self, path: str) -> bool:
        return bool(LIB.drive_mcmc_load_transitions_csv(self.handle, path.encode("utf-8")))

    def simulate(
        self,
        field_pos: int = 25,
        down: int = 1,
        yards_to_go: int = 10,
        score_differential: int = 0,
        quarter: int = 1,
    ) -> dict[str, float]:
        """
        score_differential is the possessing team's margin (positive =
        leading), fixed for the simulated drive. quarter is 1-4 (OT clamps
        to 4 on the C++ side).
        """
        p_td = ctypes.c_float()
        p_fg = ctypes.c_float()
        p_punt = ctypes.c_float()
        p_turnover = ctypes.c_float()
        yards = ctypes.c_float()
        expected_plays = ctypes.c_float()
        pass_rate = ctypes.c_float()
        value = ctypes.c_float()
        LIB.drive_mcmc_simulate(
            self.handle,
            field_pos,
            down,
            yards_to_go,
            score_differential,
            quarter,
            ctypes.byref(p_td),
            ctypes.byref(p_fg),
            ctypes.byref(p_punt),
            ctypes.byref(p_turnover),
            ctypes.byref(yards),
            ctypes.byref(expected_plays),
            ctypes.byref(pass_rate),
            ctypes.byref(value),
        )
        return {
            "p_touchdown": float(p_td.value),
            "p_field_goal": float(p_fg.value),
            "p_punt": float(p_punt.value),
            "p_turnover": float(p_turnover.value),
            "expected_yards": float(yards.value),
            "expected_plays": float(expected_plays.value),
            "expected_pass_rate": float(pass_rate.value),
            "drive_value": float(value.value),
        }
