import numpy as np


def brinson(w, b, rp, rb):
    """Allocation, selection and interaction for each segment in one period."""
    w, b, rp, rb = map(np.asarray, (w, b, rp, rb))
    Rb = b @ rb
    return (w - b) * (rb - Rb), b * (rp - rb), (w - b) * (rp - rb)


def carino(Rp, Rb):
    """Factors k_t / k that make linked effects add to the cumulative active return."""
    Rp, Rb = np.asarray(Rp, float), np.asarray(Rb, float)
    d = Rp - Rb
    kt = np.where(np.abs(d) > 1e-15,
                  (np.log1p(Rp) - np.log1p(Rb)) / np.where(d == 0, 1, d), 1 / (1 + Rp))
    P, B = np.prod(1 + Rp) - 1, np.prod(1 + Rb) - 1
    k = (np.log1p(P) - np.log1p(B)) / (P - B) if abs(P - B) > 1e-15 else 1 / (1 + P)
    return kt / k, P, B


if __name__ == "__main__":
    a, s, i = brinson([0.7, 0.3], [0.6, 0.4], [0.12, 0.01], [0.10, 0.02])
    print(f"allocation {100 * a.sum():.2f}, selection {100 * s.sum():.2f}, "
          f"interaction {100 * i.sum():.2f} points")
    k, P, B = carino([0.02, 0.03], [0.01, 0.01])
    linked = 100 * (k * np.array([0.01, 0.02])).sum()
    print(f"linked: {linked:.4f} = {100 * (P - B):.4f} points, "
          f"k = {k.round(4)}")
