from __future__ import annotations

import time

from experiments import (
    e0_verify_datasets,
    e1_similarity_threshold,
    e2_fs_vs_linear,
    e3_blocking,
    e4_anomalies_imputation,
    e5_quality_before_after,
    e6_comparison,
    e7_diagnostics,
    e8_active_learning,
    e9_robustness,
    e10_sensitivity,
    e11_duplicate_share,
    figures,
)

STEPS = (
    ("e0: перевірка наборів", e0_verify_datasets.main),
    ("e1: метрики подібності й поріг (П1)", e1_similarity_threshold.main),
    ("e2: імовірнісні моделі проти лінійної (П2)", e2_fs_vs_linear.main),
    ("e3: схеми блокування (П3)", e3_blocking.main),
    ("e4: викиди й пропуски (П4)", e4_anomalies_imputation.main),
    ("e5: індекс якості до і після (П5)", e5_quality_before_after.main),
    ("e6: порівняння з опублікованими результатами", e6_comparison.main),
    ("e7: діагностика втрат якості", e7_diagnostics.main),
    ("e8: навчання на відповідях людини", e8_active_learning.main),
    ("e9: стійкість до несхожих джерел", e9_robustness.main),
    ("e10: чутливість до параметрів", e10_sensitivity.main),
    ("e11: частка дублікатів на тому самому наборі", e11_duplicate_share.main),
    ("графіки", figures.main),
)


def main() -> None:
    started = time.perf_counter()
    for title, step in STEPS:
        print(f"\n=== {title} ===")
        step_started = time.perf_counter()
        step()
        print(f"--- {title}: {time.perf_counter() - step_started:.0f} с")
    print(f"\nУсі експерименти: {time.perf_counter() - started:.0f} с")


if __name__ == "__main__":
    main()
