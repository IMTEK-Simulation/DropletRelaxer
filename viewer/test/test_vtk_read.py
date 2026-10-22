"""The run file opens in VTK's own reader (what ParaView uses)."""
import os
import sys

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test"))

from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

from DropletRelaxer.driver import Driver  # noqa: E402
from DropletRelaxer.io.writer import H5Writer  # noqa: E402
from test_driver import Recorder, short_cfg  # noqa: E402


def test_vtkhdf_reader_sees_every_frame(tmp_path):
    path = str(tmp_path / "run.h5")
    rec = Recorder()
    Driver(short_cfg(), sinks=[rec, H5Writer(path)]).run()
    r = vtk.vtkHDFReader()
    r.SetFileName(path)
    r.UpdateInformation()
    assert r.GetNumberOfSteps() == len(rec.frames)
    for k, f in enumerate(rec.frames):
        r.SetStep(k)
        r.Update()
        out = r.GetOutput()
        assert np.array_equal(vtk_to_numpy(out.GetPoints().GetData()), f.verts)
        assert np.array_equal(vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3), f.tris)
        assert np.array_equal(vtk_to_numpy(out.GetPointData().GetArray("mean_curvature")), f.curvature,
                              equal_nan=True)
        assert out.GetFieldData().GetArray("energy").GetValue(0) == f.energy
        assert out.GetFieldData().GetArray("force").GetNumberOfComponents() == 3
