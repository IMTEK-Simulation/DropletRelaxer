"""Video export of a replay and the quasi-static frame table."""

import os

FRAME_KEYS = ["x", "y", "z", "iteration", "step_iters", "converged", "energy", "Fx", "Fy", "Fz"]


def record_video(scene, path, n_frames, show_index):
    """Renders the recorded frames one after another into a video file (the
    window content incl. charts and the info block). `show_index(k)` puts frame
    k on screen. At most ViewerConfig.video_max_frames frames are rendered
    (evenly subsampled, the last frame always included). Writes an mp4 via
    imageio-ffmpeg, or a .gif next to `path` without ffmpeg. Returns the path
    written."""
    import imageio
    vcfg, plotter = scene.vcfg, scene.plotter
    stride = max(1, -(-n_frames // max(int(vcfg.video_max_frames), 1)))
    idxs = list(range(0, n_frames, stride))
    if idxs[-1] != n_frames - 1:
        idxs.append(n_frames - 1)
    try:
        writer = imageio.get_writer(path, fps=vcfg.video_fps, codec="libx264",
                                    quality=8, macro_block_size=2)
    except Exception as err:
        path = os.path.splitext(path)[0] + ".gif"
        print(f"No ffmpeg video writer ({err}); writing {os.path.basename(path)} instead")
        writer = imageio.get_writer(path, fps=vcfg.video_fps)
    try:
        for n, k in enumerate(idxs):
            show_index(k)
            plotter.render()
            writer.append_data(plotter.screenshot(return_img=True))
            if n % max(1, len(idxs) // 10) == 0:
                print(f"  video: {100 * n // len(idxs)} %")
    finally:
        writer.close()
    return path


def write_frame_table(path, series):
    """frames.csv of a quasi-static run: one row per equilibrium frame with tip
    position x,y,z, iteration, iterations used by the step, converged flag
    (1/0), energy and the force of the liquid on the sphere Fx, Fy, Fz."""
    s = series.arrays()
    with open(path, "w") as f:
        f.write(",".join(FRAME_KEYS) + "\n")
        for k in range(len(series)):
            c, F = s["sphere_center"][k], s["force"][k]
            row = [c[0], c[1], c[2], int(s["iteration"][k]), int(s["step_iters"][k]),
                   int(bool(s["converged"][k])), s["energy"][k], F[0], F[1], F[2]]
            f.write(",".join(str(v) for v in row) + "\n")
