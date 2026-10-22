"""DropletViewer: PyVista viewer for DropletRelaxer runs (live window, run
files, video export)."""

try:
    from ._version import __version__
except ImportError:  # running from a source tree without a build
    __version__ = "0+unknown"
