"""Command line: `droplet-view`.

    droplet-view run.h5                    replay a run (history slider, Play, Save)
    droplet-view run.h5 --follow           watch a run that is still being written
    droplet-view run.h5 --video out.mp4    render the replay into a video, no window
    droplet-view live scenario.toml        run the solver in this process with a live window

--software-gl selects Mesa's software OpenGL (llvmpipe) before VTK is
loaded: slower, but it works when the GPU driver is not usable.
"""

import argparse
import os
import sys

from . import __version__


def _display_hints(software_gl):
    """Warnings for the usual reasons a VTK window fails on Linux; must run
    before VTK is imported (the software-GL variables are read at load time)."""
    if software_gl:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["GALLIUM_DRIVER"] = "llvmpipe"
        print("Using software OpenGL (Mesa llvmpipe).")
    if not sys.platform.startswith("linux"):
        return
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("Hint: DISPLAY is not set -- no X11 session reachable from this shell. "
              "For a video without a window use --video.", file=sys.stderr)
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        print("Hint: Wayland session. If the window fails to open, log in with an Xorg session "
              "or try --software-gl.", file=sys.stderr)
    if os.environ.get("CONDA_PREFIX") and os.environ.get("LD_LIBRARY_PATH"):
        print("Hint: an active Conda environment sets LD_LIBRARY_PATH, which can shadow the system "
              "OpenGL libraries (\"Could not find a decent config\"). See the troubleshooting section "
              "of the documentation.", file=sys.stderr)


def build_parser():
    p = argparse.ArgumentParser(prog="droplet-view", description="View DropletRelaxer runs.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("target", help="run file (.h5), or 'live' followed by a scenario TOML file")
    p.add_argument("config", nargs="?", help="with 'live': the scenario TOML file (default: all defaults)")
    p.add_argument("--follow", action="store_true", help="keep up with a run that is still being written")
    p.add_argument("--video", metavar="OUT", help="render the replay into a video file instead of opening a window")
    p.add_argument("-o", "--output", help="with 'live': run file to write (default runs/<timestamp>/run.h5)")
    p.add_argument("--force", action="store_true", help="with 'live': overwrite an existing output file")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value; with a run file only viewer.* (repeatable)")
    p.add_argument("--software-gl", action="store_true", help="use Mesa software OpenGL (slow, robust)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    _display_hints(args.software_gl and not args.video)
    from DropletRelaxer.config import ConfigError
    try:
        from . import app
        if args.target == "live":
            app.live(args.config, args.set, output=args.output, force=args.force)
            return 0
        if args.config:
            raise SystemExit("a second positional argument is only allowed with 'live'")
        viewer_sets = []
        for s in args.set:
            if not s.startswith("viewer."):
                raise ConfigError(f"'{s}': only viewer.* can be set when viewing a run file")
            viewer_sets.append(s[len("viewer."):])
        if args.video:
            print(f"Video written to {app.export_video(args.target, args.video, viewer_sets)}")
        else:
            app.replay(args.target, viewer_sets, follow=args.follow)
    except ConfigError as err:
        print(f"droplet-view: configuration error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
