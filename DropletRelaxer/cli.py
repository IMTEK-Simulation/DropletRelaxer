"""Command line: `droplet-relax`.

    droplet-relax run scenario.toml -o run.h5 [--set solver.contact_angle_deg=120 ...]
    droplet-relax restart run.h5 [--set solver.n_iterations=2000000]
    droplet-relax info run.h5
    droplet-relax show-config [scenario.toml] [--set ...]

SIGTERM (sent by SLURM before a wall-time kill) and the first Ctrl-C end a
run cleanly: the current iteration finishes, the final frame and the restart
checkpoint are written. A second Ctrl-C aborts immediately.
"""

import argparse
import os
import signal
import sys

from . import __version__
from .config import ConfigError, dumps, load, loads
from .console import ConsoleSink
from .driver import Driver
from .io.frame import StopRequest


def install_signal_handlers(stop):
    def _graceful(signum, _frame):
        name = signal.Signals(signum).name
        if signum == signal.SIGINT and stop.requested:
            raise KeyboardInterrupt
        stop.request(name)
        print(f"\n{name}: finishing the current iteration and writing the checkpoint "
              "(Ctrl-C again to abort)", file=sys.stderr, flush=True)
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)


def _read_config(path, overrides):
    if path is None:
        return loads("", overrides)
    return load(path, overrides)


def cmd_run(args):
    cfg, table = _read_config(args.config, args.set)
    sinks = [ConsoleSink(progress=not args.quiet and sys.stdout.isatty())]
    writer = None
    if args.output:
        from .io.writer import H5Writer
        if os.path.exists(args.output) and not args.force:
            raise SystemExit(f"{args.output} exists; use --force to overwrite it")
        writer = H5Writer(args.output, viewer_table=table.get("viewer"), command_line=sys.argv)
        sinks.append(writer)
    stop = StopRequest()
    install_signal_handlers(stop)
    Driver(cfg, sinks=sinks, stop=stop).run()
    if writer is not None:
        print(f"Run written to {os.path.abspath(args.output)}")
    return 0


def cmd_restart(args):
    from .io.reader import RunFile
    from .io.writer import H5Writer
    with RunFile(args.run) as rf:
        cfg, table = rf.config(args.set)
        checkpoint = rf.checkpoint()
        viewer_table = table.get("viewer")
    if checkpoint is None:
        raise SystemExit(f"{args.run} has no checkpoint to restart from")
    stop = StopRequest()
    install_signal_handlers(stop)
    sinks = [ConsoleSink(progress=not args.quiet and sys.stdout.isatty()),
             H5Writer(args.run, viewer_table=viewer_table, command_line=sys.argv, append=True)]
    Driver(cfg, sinks=sinks, stop=stop, restart=checkpoint).run()
    return 0


def cmd_info(args):
    from .io.reader import RunFile
    with RunFile(args.run) as rf:
        print(rf.describe())
    return 0


def cmd_show_config(args):
    cfg, table = _read_config(args.config, args.set)
    sys.stdout.write(dumps(cfg, viewer=table.get("viewer"), docs=not args.no_docs))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="droplet-relax", description="Relax a liquid droplet on a substrate.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_set(sp):
        sp.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value (TOML syntax; repeatable)")

    r = sub.add_parser("run", help="run a scenario")
    r.add_argument("config", nargs="?", help="scenario TOML file (default: all defaults)")
    r.add_argument("-o", "--output", help="run file to write (HDF5, VTKHDF layout)")
    r.add_argument("--force", action="store_true", help="overwrite an existing output file")
    r.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    add_set(r)
    r.set_defaults(func=cmd_run)

    rs = sub.add_parser("restart", help="continue a run from the checkpoint in its file")
    rs.add_argument("run", help="run file written by `droplet-relax run`")
    rs.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    rs.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="only solver.n_iterations and output.* may change at a restart")
    rs.set_defaults(func=cmd_restart)

    i = sub.add_parser("info", help="summary of a run file")
    i.add_argument("run")
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("show-config", help="print the complete configuration with documentation")
    s.add_argument("config", nargs="?", help="scenario TOML file (default: all defaults)")
    s.add_argument("--no-docs", action="store_true", help="values only, no comments")
    add_set(s)
    s.set_defaults(func=cmd_show_config)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as err:
        print(f"droplet-relax: configuration error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
