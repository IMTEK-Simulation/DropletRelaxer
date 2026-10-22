"""Mouse interaction: right-drag pans the camera (instead of VTK's dolly)."""

import vtk


class _RightDragPanStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Right-click-drag pans (translates) the camera instead of
    zooming it (the vtkInteractorStyleTrackballCamera default). Left-
    click-drag still rotates and the scroll wheel still zooms -- only
    the right button's action changes. Handy since the default pan
    button (middle-click) has no convenient equivalent on most
    trackpads.

    Implemented by overriding OnRightButtonDown/OnRightButtonUp
    directly. An earlier version used AddObserver("RightButtonPressEvent",
    ...) instead, on the theory that a direct override wasn't reliably
    dispatched -- that was wrong and was the actual bug: VTK's own
    vtkInteractorStyleTrackballCamera.OnRightButtonDown() override does
    NOT invoke the "RightButtonPressEvent" VTK event at all (only the
    common base class's default implementation does, which the
    trackball-camera subclass replaces rather than calls), so that
    observer silently never fired and every right-click fell straight
    through to the default Dolly/zoom -- confirmed by directly calling
    OnRightButtonDown() on both versions and checking GetState()
    (VTKIS_DOLLY vs. the intended VTKIS_PAN). The direct override below
    does not call the inherited OnRightButtonDown()/Up() (doing so
    would also run the default StartDolly()/EndDolly(), fighting our
    pan) and is the standard, verified-working pattern."""

    def OnRightButtonDown(self):
        # Same preamble as VTK's own OnRightButtonDown: the renderer under
        # the cursor must be made the current one, otherwise Pan() has no
        # camera to move and silently does nothing (this was the "right
        # drag no longer works" bug once the window contained several
        # renderers/charts).
        x, y = self.GetInteractor().GetEventPosition()
        self.FindPokedRenderer(x, y)
        if self.GetCurrentRenderer() is None:
            return
        self.StartPan()

    def OnRightButtonUp(self):
        self.EndPan()


def enable_right_button_pan(plotter):
    """Call once, after the interactive window is open (i.e. after
    plotter.show(...)), to remap the right mouse button as described
    in _RightDragPanStyle.

    Sets it through PyVista's OWN tracked `plotter.iren.style` property
    rather than calling `.interactor.SetInteractorStyle(...)` directly.
    The latter looks equivalent and works at first, but every
    `plotter.show()` call -- including the SECOND one after the history
    slider is added at the end of the run -- internally calls
    `iren.update_style()`, which reapplies whatever PyVista itself has
    tracked as the current style, silently reverting a style that was
    only ever set on the raw interactor. Setting `iren.style` instead
    makes PyVista track OUR style, so it survives every later
    `update_style()`/`show()` call.

    That alone turned out not to be enough in practice: right-drag pan
    was reported broken again, both mid-run and after the slider
    appears. PyVista has MORE than one path that can reassign the
    interactor's style out from under a plain one-time assignment --
    e.g. double-clicking a chart (the energy chart docked at the top of
    this window covers a good chunk of the screen) toggles a
    Context/chart-interaction style via `_set_context_style`, and other
    widgets can have similar effects -- and chasing down every possible
    trigger individually is fragile. Instead this installs a self-
    healing 'RenderEvent' observer that re-applies our style on every
    single render if something else has swapped it out, which is cheap
    (one identity check per render) and correct regardless of WHAT
    caused the swap."""
    style = getattr(plotter, "_right_drag_pan_style", None)
    if style is None:
        style = _RightDragPanStyle()
        plotter._right_drag_pan_style = style

        def _reassert_style(*_args):
            if plotter.iren.interactor.GetInteractorStyle() is not style:
                plotter.iren.style = style

        plotter.iren.add_observer("RenderEvent", _reassert_style)

        # The Python overrides OnRightButtonDown/Up above are NOT called by
        # VTK's C++ event dispatch (measured: after a real right-button press
        # the style was in VTKIS_DOLLY, not PAN -- the C++ implementation of
        # vtkInteractorStyleTrackballCamera runs instead). So the press/release
        # are intercepted on the interactor with a higher priority than the
        # style's own handler, start/stop the pan there, and abort the event so
        # the default dolly never starts.
        inter = plotter.iren.interactor

        def _make(handler):
            holder = {}

            def cb(_obj, _event):
                if handler():
                    holder["cmd"].SetAbortFlag(1)
            tag = inter.AddObserver(handler.event, cb, 10.0)
            holder["cmd"] = inter.GetCommand(tag)
            return tag

        def _press():
            cur = plotter.iren.interactor.GetInteractorStyle()
            x, y = inter.GetEventPosition()
            cur.FindPokedRenderer(x, y)
            if cur.GetCurrentRenderer() is None:
                return False
            cur.StartPan()
            return True

        def _release():
            cur = plotter.iren.interactor.GetInteractorStyle()
            if cur.GetState() == 2:  # VTKIS_PAN
                cur.EndPan()
                return True
            return False

        _press.event = "RightButtonPressEvent"
        _release.event = "RightButtonReleaseEvent"
        plotter._right_drag_tags = (_make(_press), _make(_release))

    plotter.iren.style = style
