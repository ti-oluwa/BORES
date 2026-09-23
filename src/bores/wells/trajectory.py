"""
Well trajectory (deviation survey) representation.

A `WellTrajectory` is an ordered set of `(measured_depth, x, y, z)`
stations, piecewise-linear between consecutive stations. A raw survey
polyline, not a minimum-curvature-interpolated one. Accuracy depends on
station spacing, same as it would for any survey-based trajectory tool.
"""

import math

import attrs

from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.types import Number

__all__ = ["TrajectoryStation", "WellTrajectory"]


@attrs.frozen(kw_only=True, slots=True)
class TrajectoryStation(Serializable):
    """One survey station: position along the wellbore at a given measured depth."""

    measured_depth: Number
    """
    Distance along the wellbore from its measured-depth origin (usually
    the rotary table / surface reference point).
    """
    x: Number
    y: Number
    z: Number
    """
    True vertical depth, positive-down (same convention as
    `Perforation.top_depth` and `Grid.vertex_coordinates`).
    """


@attrs.frozen(kw_only=True, slots=True)
class WellTrajectory(Serializable):
    """
    A well's 3-D path: an ordered, piecewise-linear polyline of `TrajectoryStation`.

    `Perforation` intervals on a well with this trajectory are defined
    against its measured depth, not true vertical depth, as TVD is not
    invertible along a horizontal or S-shaped section (multiple measured
    depths can share the same TVD), so measured depth is the only interval
    representation that identifies a unique location on an arbitrary path.
    """

    stations: tuple[TrajectoryStation, ...] = attrs.field(converter=tuple)

    def __attrs_post_init__(self) -> None:
        if len(self.stations) < 2:
            raise ValidationError(
                f"`{type(self).__name__}` needs at least 2 stations to define a "
                f"path; got {len(self.stations)}."
            )
        for previous, current in zip(self.stations, self.stations[1:], strict=False):
            if current.measured_depth <= previous.measured_depth:
                raise ValidationError(
                    "`stations` must be strictly increasing in "
                    f"`measured_depth`; {previous.measured_depth} is "
                    f"followed by {current.measured_depth}."
                )

    @property
    def top_measured_depth(self) -> Number:
        """Measured depth of the first station."""
        return self.stations[0].measured_depth

    @property
    def bottom_measured_depth(self) -> Number:
        """Measured depth of the last station."""
        return self.stations[-1].measured_depth

    def get_bracketing_leg(
        self, measured_depth: Number
    ) -> tuple[TrajectoryStation, TrajectoryStation]:
        """Returns the two consecutive stations whose measured-depth range contains `measured_depth`."""
        if not (self.top_measured_depth <= measured_depth <= self.bottom_measured_depth):
            raise ValidationError(
                f"measured_depth={measured_depth} is outside this "
                f"trajectory's range [{self.top_measured_depth}, "
                f"{self.bottom_measured_depth}]."
            )
        for previous, current in zip(self.stations, self.stations[1:], strict=False):
            if previous.measured_depth <= measured_depth <= current.measured_depth:
                return previous, current
        return self.stations[-2], self.stations[-1]

    def stations_between(self, start_md: Number, end_md: Number) -> tuple[TrajectoryStation, ...]:
        """
        Returns every vertex on the polyline between `start_md` and `end_md`,
        inclusive - the interpolated position at `start_md`, every real
        survey station strictly between them, and the interpolated
        position at `end_md`.

        Returns two entries (just the endpoints) if no real
        station falls strictly inside the range.

        :param start_md: Range start, `<= end_md`.
        :param end_md: Range end, `>= start_md`.
        :returns: Ordered tuple of `TrajectoryStation`, length >= 2.
        :raises ValidationError: If `start_md > end_md`, or either falls
            outside this trajectory's range.
        """
        if start_md > end_md:
            raise ValidationError(f"`start_md` ({start_md}) must be <= `end_md` ({end_md}).")

        interior = tuple(
            station for station in self.stations if start_md < station.measured_depth < end_md
        )
        start_x, start_y, start_z = self.position_at(start_md)
        end_x, end_y, end_z = self.position_at(end_md)
        return (
            TrajectoryStation(measured_depth=start_md, x=start_x, y=start_y, z=start_z),
            *interior,
            TrajectoryStation(measured_depth=end_md, x=end_x, y=end_y, z=end_z),
        )

    def position_at(self, measured_depth: Number) -> tuple[Number, Number, Number]:
        """
        Returns a linearly-interpolated `(x, y, z)` at `measured_depth`.

        :param measured_depth: Must be within `[top_measured_depth, bottom_measured_depth]`.
        :returns: `(x, y, z)`.
        :raises ValidationError: If `measured_depth` is outside range.
        """
        previous, current = self.get_bracketing_leg(measured_depth)
        span = current.measured_depth - previous.measured_depth
        fraction = 0.0 if span == 0 else (measured_depth - previous.measured_depth) / span
        return (
            previous.x + fraction * (current.x - previous.x),
            previous.y + fraction * (current.y - previous.y),
            previous.z + fraction * (current.z - previous.z),
        )

    def tangent_at(self, measured_depth: Number) -> tuple[Number, Number, Number]:
        """
        Returns the unit tangent vector at `measured_depth`. The direction of the leg
        containing it. Constant within a leg (piecewise-linear trajectory).

        :returns: Unit vector `(dx, dy, dz)`.
        :raises ValidationError: If the bracketing leg has zero length.
        """
        previous, current = self.get_bracketing_leg(measured_depth)
        dx = current.x - previous.x
        dy = current.y - previous.y
        dz = current.z - previous.z
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        if length == 0:
            raise ValidationError(
                "Zero-length trajectory leg between measured_depth="
                f"{previous.measured_depth} and {current.measured_depth}."
            )
        return dx / length, dy / length, dz / length

    def inclination_at(self, measured_depth: Number) -> Number:
        """
        Returns the angle between the local tangent and vertical (+z, positive-down),
        in radians. `0` for a vertical-and-descending trajectory, `pi/2`
        for horizontal, matching `inclination_from_vertical` everywhere
        else in `wells`.

        :returns: Inclination in `[0, pi]`.
        """
        _, _, tangent_z = self.tangent_at(measured_depth)
        return math.acos(max(-1.0, min(1.0, tangent_z)))
