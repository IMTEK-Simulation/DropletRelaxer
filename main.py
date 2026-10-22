"""
Development entry point: runs the default Config() with the live window.

    python main.py

The run loop lives in DropletRelaxer.driver, the window in
DropletViewer.live; this script only wires them together.
"""

import os
import time

from DropletRelaxer.config import Config
from DropletRelaxer.console import ConsoleSink
from DropletRelaxer.driver import Driver
from DropletViewer.config import ViewerConfig
from DropletViewer.live import LiveWindow


def main(cfg=None, vcfg=None):
    cfg = Config() if cfg is None else cfg
    save_dir = os.path.join("runs", time.strftime("droplet_run_%Y%m%d_%H%M%S"))
    sinks = [ConsoleSink(), LiveWindow(vcfg if vcfg is not None else ViewerConfig(), save_dir=save_dir)]
    return Driver(cfg, sinks=sinks).run()


if __name__ == "__main__":
    main()
