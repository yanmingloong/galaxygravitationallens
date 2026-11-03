"""Thin orchestration layer: run a sequence of stages, or call each stage by hand.

This is the "auto" layer on top of the pluggable components. The components
(``gl.fit``, ``selection.run_selection``, ``skirt.*``, ``validation.*``, ...)
stay callable on their own -- the manual mode. ``Pipeline`` just wires them in
order, so an end-to-end run is one call while every stage remains individually
reachable.

Design rule (the repo's "engine + handle" split): a ``Stage`` wraps ONE
component call; a ``Pipeline`` is an ordered list of stages plus a ``ctx`` dict
carried stage-to-stage. ``run`` supports three granularities:

- full auto:      ``pipeline.run()``
- partial:        ``pipeline.run(only=["fit", "analyze"])``
- range:          ``pipeline.run(start="fit", stop="plot")``

Manual mode does not go through ``Pipeline`` at all -- call the underlying
functions directly, exactly as before this layer existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class Stage:
    """One pipeline step: a named callable that reads/writes ``ctx``."""

    name: str
    fn: Callable[[Dict[str, Any]], Any]
    doc: str = ""
    depends: Sequence[str] = ()  # stage names this stage needs to have run

    def __call__(self, ctx: Dict[str, Any]) -> Any:
        out = self.fn(ctx)
        if out is not None:
            ctx[self.name] = out
        return out


class Pipeline:
    """An ordered list of :class:`Stage` executed against a shared ``ctx``."""

    def __init__(self, name: str, stages: Sequence[Stage]):
        self.name = name
        self._stages: Dict[str, Stage] = {}
        self._order: List[str] = []
        for s in stages:
            if s.name in self._stages:
                raise ValueError(f"duplicate stage name: {s.name!r}")
            self._stages[s.name] = s
            self._order.append(s.name)

    @property
    def stage_names(self) -> List[str]:
        return list(self._order)

    def _select(self, start=None, stop=None, only=None) -> List[str]:
        """Resolve start/stop/only into an ordered list of stage names."""
        names = self._order
        if only is not None:
            unknown = [n for n in only if n not in self._stages]
            if unknown:
                raise KeyError(f"unknown stages: {unknown}")
            return [n for n in names if n in set(only)]
        i0 = names.index(start) if start is not None else 0
        i1 = names.index(stop) + 1 if stop is not None else len(names)
        if start is not None and start not in self._stages:
            raise KeyError(f"unknown start stage: {start!r}")
        if stop is not None and stop not in self._stages:
            raise KeyError(f"unknown stop stage: {stop!r}")
        return names[i0:i1]

    def run(
        self,
        ctx: Optional[Dict[str, Any]] = None,
        start=None,
        stop=None,
        only=None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the selected stages in order against ``ctx`` (created if None).

        ``only`` takes precedence over ``start``/``stop``; stages keep their
        declared order regardless of the selection order.
        """
        ctx = {} if ctx is None else ctx
        for name in self._select(start=start, stop=stop, only=only):
            stage = self._stages[name]
            missing = [d for d in stage.depends if d not in ctx]
            if missing:
                raise RuntimeError(
                    f"stage {name!r} needs {missing} in ctx; run them first"
                )
            if verbose:
                print(f"[{self.name}] {name}" + (f" -- {stage.doc}" if stage.doc else ""))
            stage(ctx)
        return ctx


def make_pipeline(name: str, stages: Sequence[Stage]) -> Pipeline:
    """Convenience constructor (alias of ``Pipeline``)."""
    return Pipeline(name, stages)
