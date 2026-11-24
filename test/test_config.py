"""TOML configuration: strict loading, overrides, round trip, documentation."""
import glob
import os
import subprocess
import sys
from dataclasses import fields, is_dataclass

import pytest

from DropletRelaxer.config import Config, ConfigError, apply_volume_scale, dumps, field_docs, loads

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_empty_file_gives_defaults():
    cfg, table = loads("")
    assert cfg == Config() and table == {}


def test_round_trip_is_exact():
    cfg = Config()
    cfg.solver.contact_angle_deg = 117.3
    cfg.obstacle.center_end = (12.5, -1.0, 130.0)
    cfg.roughness.rms_height = 0.1 + 0.2       # a float that needs all 17 digits
    cfg2, _ = loads(dumps(cfg))
    assert cfg2 == cfg
    assert loads(dumps(cfg, docs=True))[0] == cfg


def test_types_are_checked_and_converted():
    cfg, _ = loads("[solver]\nlearning_rate = 1\n[obstacle]\ncenter_start = [0, 0, 125]\n")
    assert isinstance(cfg.solver.learning_rate, float)
    assert cfg.obstacle.center_start == (0.0, 0.0, 125.0)
    for bad in ("[solver]\nn_iterations = 1.5\n", "[solver]\nn_iterations = true\n",
                "[obstacle]\ncenter_start = [0, 0]\n", "[mesh]\nmesh_type = 3\n", "solver = 3\n"):
        with pytest.raises(ConfigError):
            loads(bad)


def test_unknown_keys_are_errors_with_suggestion():
    with pytest.raises(ConfigError, match="did you mean 'contact_angle_deg'"):
        loads("[solver]\ncontact_angel_deg = 90\n")
    with pytest.raises(ConfigError, match="did you mean 'solver'"):
        loads("[solvr]\n")


def test_overrides():
    cfg, table = loads("[solver]\ncontact_angle_deg = 60.0\n",
                       ["solver.contact_angle_deg=120", "mesh.mesh_type=polar", "obstacle.center_end=[1, 2, 3]",
                        "viewer.colormap=viridis"])
    assert cfg.solver.contact_angle_deg == 120.0
    assert cfg.mesh.mesh_type == "polar"            # bare word taken as a string
    assert cfg.obstacle.center_end == (1.0, 2.0, 3.0)
    assert table["viewer"] == {"colormap": "viridis"}
    with pytest.raises(ConfigError):
        loads("", ["solver.contact_angle_deg"])


def test_viewer_table_is_ignored_by_the_solver():
    cfg, table = loads("[viewer]\nanything = 1\n")
    assert cfg == Config() and table["viewer"] == {"anything": 1}


def test_every_field_is_documented():
    missing = {}
    for group in fields(Config):
        cls = group.type
        assert is_dataclass(cls)
        docs = field_docs(cls)
        missing[group.name] = [f.name for f in fields(cls) if not docs.get(f.name)]
    assert not any(missing.values()), f"fields without docstring: {missing}"


def test_apply_volume_scale_returns_a_scaled_copy():
    cfg = Config()
    s = apply_volume_scale(cfg, 2.0)
    assert s.solver.volume_penalty == cfg.solver.volume_penalty / 16
    assert s.obstacle.capture_radius == 2 * cfg.obstacle.capture_radius
    assert cfg == Config()                          # the input is untouched
    assert apply_volume_scale(cfg, 1.0) == cfg


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "examples", "*.toml"))))
def test_examples_load(path):
    with open(path) as f:
        loads(f.read())


def test_cli_show_config_and_errors():
    out = subprocess.run([sys.executable, "-m", "DropletRelaxer.cli", "show-config", "--no-docs",
                          "--set", "solver.contact_angle_deg=33"], capture_output=True, text=True, check=True)
    assert loads(out.stdout)[0].solver.contact_angle_deg == 33.0
    bad = subprocess.run([sys.executable, "-m", "DropletRelaxer.cli", "show-config", "--set", "solver.nope=1"],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown key" in bad.stderr


@pytest.mark.parametrize("command", [["run"], ["restart", "run.vtkhdf"]])
def test_cli_native_kernels_is_scoped(command, monkeypatch):
    from DropletRelaxer import cli, _fast

    def run(args):
        assert args.native_kernels and _fast.use_native_kernels()
        return 0

    monkeypatch.setattr(_fast, "numba_available", lambda: True)
    monkeypatch.setattr(_fast, "_native_requested", False)
    monkeypatch.setattr(cli, "cmd_" + command[0], run)
    assert cli.main([*command, "--native-kernels"]) == 0
    assert not _fast.use_native_kernels()


@pytest.mark.parametrize("mode", ["flag", "environment", "broken-numba", "old-numba"])
def test_cli_can_run_without_loading_kernels(mode):
    code = '''
import importlib.util
import os
import sys
from types import SimpleNamespace

mode = sys.argv[1]
os.environ['DROPLET_NATIVE_KERNELS'] = '1' if mode == 'environment' else '0'

class CheckBackend:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numba' and mode == 'broken-numba':
            return importlib.util.spec_from_loader(fullname, self)
        if fullname in ('numba', 'DropletRelaxer.kernels'):
            raise SystemExit(f'unexpected attempt to load {fullname}')

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise RuntimeError('incompatible llvmlite')

sys.meta_path.insert(0, CheckBackend())
if mode == 'old-numba':
    sys.modules['numba'] = SimpleNamespace(njit=lambda: None, version_info=(0, 62))

from DropletRelaxer.cli import main

args = ['run', '-q', '--set', 'solver.n_iterations=15', '--set', 'anomaly.enabled=true']
if mode == 'flag':
    args.append('--native-kernels')
raise SystemExit(main(args))
'''
    subprocess.run([sys.executable, "-B", "-c", code, mode], check=True, timeout=30)
