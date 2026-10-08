"""Feature blocks. Importing this package registers every block.

One module per family; see :mod:`taxiout.features.registry` for the declaration rules. This
package holds the 24 blocks that the members of the best submission (v62) read, plus `snow`
(final phase, 2026-10-08).
"""

from taxiout.features import (  # noqa: F401
    adsb,
    adsb_calib,
    arrival_regime,
    calendar,
    congestion,
    consistency,
    context_window,
    eurocontrol_daily,
    flight_match,
    gate_retention,
    modes,
    modes5,
    modes6,
    offblock,
    opdi,
    queue2,
    runway_config,
    runway_history,
    schedule,
    snow,
    stand_geometry,
    stand_runway,
    stand_runway_lo,
    taxi_in,
    weather,
)
from taxiout.features.registry import (
    REGISTRY,
    FeatureBlock,
    FeatureContext,
    FeatureKind,
    align,
    feature,
    get,
    resolve,
)

__all__ = [
    "REGISTRY",
    "FeatureBlock",
    "FeatureContext",
    "FeatureKind",
    "align",
    "feature",
    "get",
    "resolve",
]
