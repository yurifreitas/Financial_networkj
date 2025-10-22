import numpy as np
import pywt

def compute(series: np.ndarray, wavelet: str = "db4", level: int = 4) -> float:
    """Energia média das componentes wavelet."""
    try:
        coeffs = pywt.wavedec(series, wavelet, level=level)
        energies = [np.mean(np.square(c)) for c in coeffs]
        return np.sum(energies)
    except Exception:
        return np.nan
