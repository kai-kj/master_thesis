from distinctipy import get_colors


def palette(
        size: int, index: int,
        pastel_factor: float = 0.0,
        darken_factor: float = 0.0,
        colorblind: bool = True,
) -> tuple[float, float, float]:
    d_color = get_colors(
        size,
        pastel_factor=pastel_factor,
        colorblind_type="Deuteranomaly" if colorblind else None,
        rng=0
    )[index]
    return tuple(c * (1.0 - darken_factor) for c in d_color)  # type: ignore
