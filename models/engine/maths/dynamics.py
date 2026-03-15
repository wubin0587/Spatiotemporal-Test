# -*- coding: utf-8 -*-
# Compatibility shim — the dynamics implementation has moved to the
# dynamics/ package. This file re-exports the unified dispatcher so that
# any legacy import of maths.dynamics.calculate_opinion_change still works.

from .dynamics import calculate_opinion_change  # noqa: F401