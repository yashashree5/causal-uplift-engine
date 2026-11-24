"""Check all X5 configurations and controls without data or checkpoints."""

from itertools import combinations

import numpy as np
from configurations import DESIGN, PARTS, assemble_configuration


def main():
    expected = ["Original"] + ["".join(group) for n in range(1, 5) for group in combinations("ABCD", n)]
    assert DESIGN["core_configurations"] == expected
    assert DESIGN["hosts"] == ["causalpfn", "ptonet"]
    baseline = {part: np.arange(72, dtype=np.float32).reshape(12, 6) + i for i, part in enumerate(PARTS)}
    # A missing validation operand must use the mean of the training customers.
    baseline["validation"][0, 0] = np.nan
    components = {c: {p: np.full((12, width), i + 1, dtype=np.float32) for p in PARTS}
                  for i, (c, width) in enumerate((("A", 16), ("B", 1), ("C", 1)))}
    fixed = assemble_configuration("D", baseline)
    for design in fixed.values():
        a, b, c, d = design[:, :4].T
        np.testing.assert_allclose(design[:, 6:], np.column_stack((a+b, c-d, a*c, b/(1+abs(d)))))
    assert fixed["validation"][0, 0] == np.mean(baseline["train"][:, 0])

    for name in expected:
        blocks = "" if name == "Original" else name
        assembled = assemble_configuration(name, baseline, components)
        for part in PARTS:
            pieces = [fixed[part][:, :6]]
            for c in blocks:
                pieces.append(fixed[part][:, 6:] if c == "D" else components[c][part])
            np.testing.assert_array_equal(assembled[part], np.column_stack(pieces))

    for name in DESIGN["formula_controls"]:
        first = assemble_configuration(name, baseline)
        repeat = assemble_configuration(name, baseline)
        for i, part in enumerate(PARTS):
            assert first[part].shape == (12, 10)
            np.testing.assert_array_equal(first[part], repeat[part])
            np.testing.assert_array_equal(first[part][:, :6], fixed[part][:, :6])
            if name == "D_shuffled":
                rng = np.random.default_rng(np.random.SeedSequence([91427, i]))
                np.testing.assert_array_equal(first[part][:, 6:], fixed[part][rng.permutation(12), 6:])
    print("Verified 16 core configurations, two controls, and both hosts.")


if __name__ == "__main__":
    main()
