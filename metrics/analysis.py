import json
from typing import Dict, Any
from .entropy import shannon_entropy
from .histogram import image_histogram
from .correlation import pixel_correlation
from .npcr_uaci import npcr, uaci

def run_all(original: Any, scrambled: Any, encrypted_pkg: bytes, timings: Dict[str, float]) -> Dict[str, Any]:
    """Compute all metrics and return a serialisable dict.

    * ``original`` – original NumPy image array (uint8).
    * ``scrambled`` – image after geometric + RGB permutations (uint8).
    * ``encrypted_pkg`` – the full encrypted package bytes.
    * ``timings`` – dict with keys ``pattern``, ``rgb``, ``aes_enc``, ``total_enc`` etc.
    """
    # Basic entropy
    entropy_orig = shannon_entropy(original)
    entropy_scrambled = shannon_entropy(scrambled)
    # Histograms
    hist_orig = image_histogram(original)
    hist_scrambled = image_histogram(scrambled)
    # Correlation
    corr_orig = pixel_correlation(original)
    corr_scrambled = pixel_correlation(scrambled)
    # NPCR & UACI between original and scrambled
    npcr_val = npcr(original, scrambled)
    uaci_val = uaci(original, scrambled)
    # File sizes
    orig_size = original.nbytes
    enc_size = len(encrypted_pkg)
    overhead = enc_size - orig_size

    # Assemble dict (convert numpy arrays to lists for JSON)
    def convert_hist(hist_dict):
        return {k: v.tolist() for k, v in hist_dict.items()}

    result = {
        "entropy": {
            "original": entropy_orig,
            "scrambled": entropy_scrambled,
        },
        "histogram": {
            "original": convert_hist(hist_orig),
            "scrambled": convert_hist(hist_scrambled),
        },
        "correlation": {
            "original": corr_orig,
            "scrambled": corr_scrambled,
        },
        "npcr": npcr_val,
        "uaci": uaci_val,
        "file_size": {
            "original_bytes": orig_size,
            "encrypted_bytes": enc_size,
            "overhead_bytes": overhead,
        },
        "timing": timings,
    }
    return result
