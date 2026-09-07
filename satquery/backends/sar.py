from satquery.backends.base import BackendUnavailable


class ExperimentalSARBackend:
    """Future: sensor calibration -> speckle treatment -> encoder -> optical/SAR consistency."""

    name = "sar/interface-only"

    def analyze(self, images, question):
        raise BackendUnavailable(
            "SAR and optical/SAR fusion inference are not implemented. "
            "Raster preview and metadata are available; use a calibrated sensor-specific "
            "pipeline before interpretation."
        )
