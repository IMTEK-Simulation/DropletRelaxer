"""DropletRelaxer: surface-energy relaxation of a liquid droplet on a substrate.

Headless solver package. It must never import PyVista/VTK; the viewer lives in
the separate DropletViewer distribution.
"""

try:
    from ._version import __version__
except ImportError:  # running from a source tree without a build
    __version__ = "0+unknown"
