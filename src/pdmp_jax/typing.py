from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from jaxtyping import Array, Float, PRNGKeyArray

if TYPE_CHECKING:
    from typing import TypeAlias

Position: TypeAlias = Float[Array, " dim"]
Velocity: TypeAlias = Float[Array, " dim"]
Time: TypeAlias = Float[Array, ""]
RateIntensity: TypeAlias = Float[Array, ""]

Integrator: TypeAlias = Callable[[Position, Velocity, Time], tuple[Position, Velocity]]
RateFunction: TypeAlias = Callable[[Position, Velocity, Time], RateIntensity]
JumpFunction: TypeAlias = Callable[[Position, Velocity, PRNGKeyArray], Velocity]
