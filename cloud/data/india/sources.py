"""Indian government Antarctic data sources used by Aeolus.

The primary source is the Indian National Polar Data Center (NPDC/NCPOR).
NPDC exposes Indian Antarctic observations (including AWS and synoptic/weather
observations from Maitri and Bharati) and also catalogs Indian Antarctic and
Southern Ocean products.  Access to some datasets is request/download based,
so this module deliberately separates *source discovery* from transport.

NCMRWF is registered as a second Indian source.  Its coupled atmosphere/ocean/
sea-ice system produces Antarctic sea-ice and drift products, but the exact
machine-readable access route must be configured before a live connector is
enabled.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IndianDataSource:
    source_id: str
    provider: str
    description: str
    data_types: tuple[str, ...]
    official_url: str
    access: str


NPDC_ANTARCTIC_WEATHER = IndianDataSource(
    source_id="india.npdc.antarctic_weather",
    provider="NCPOR / Indian National Polar Data Center",
    description="Indian Antarctic station observations from Maitri and Bharati.",
    data_types=("temperature", "relative_humidity", "pressure", "wind_speed", "wind_direction"),
    official_url="https://pdc.ncpor.res.in/pdc/",
    access="request_or_export",
)

NCMRWF_ANTARCTIC = IndianDataSource(
    source_id="india.ncmrwf.antarctic_coupled",
    provider="NCMRWF",
    description="Indian coupled atmosphere/ocean/sea-ice products including Antarctic sea-ice fraction and drift.",
    data_types=("weather", "ocean", "sea_ice", "sea_ice_drift"),
    official_url="https://www.ncmrwf.gov.in/",
    access="catalog_or_configured_download",
)

INDIAN_ANTARCTIC_SOURCES = (NPDC_ANTARCTIC_WEATHER, NCMRWF_ANTARCTIC)


def source_registry() -> dict[str, IndianDataSource]:
    return {source.source_id: source for source in INDIAN_ANTARCTIC_SOURCES}
