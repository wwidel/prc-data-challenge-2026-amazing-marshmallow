"""Reference data for the ten airports in the challenge."""

from __future__ import annotations

#: ICAO -> (IATA, name, IANA timezone)
AIRPORTS: dict[str, tuple[str, str, str]] = {
    "EDDF": ("FRA", "Frankfurt Main", "Europe/Berlin"),
    "EDDM": ("MUC", "Munich", "Europe/Berlin"),
    "EGLL": ("LHR", "London Heathrow", "Europe/London"),
    "EHAM": ("AMS", "Amsterdam Schiphol", "Europe/Amsterdam"),
    "LEBL": ("BCN", "Barcelona El Prat", "Europe/Madrid"),
    "LEMD": ("MAD", "Madrid Barajas", "Europe/Madrid"),
    "LFPG": ("CDG", "Paris Charles de Gaulle", "Europe/Paris"),
    "LIRF": ("FCO", "Rome Fiumicino", "Europe/Rome"),
    "LSZH": ("ZRH", "Zurich", "Europe/Zurich"),
    "LTFM": ("IST", "Istanbul", "Europe/Istanbul"),
}

ICAO: tuple[str, ...] = tuple(AIRPORTS)
TIMEZONES: dict[str, str] = {icao: tz for icao, (_, _, tz) in AIRPORTS.items()}


def by_timezone() -> dict[str, list[str]]:
    """IANA timezone -> the ICAO codes in it."""
    grouped: dict[str, list[str]] = {}
    for icao, tz in TIMEZONES.items():
        grouped.setdefault(tz, []).append(icao)
    return grouped
