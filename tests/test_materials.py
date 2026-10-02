import numpy as np

from cardimech.materials import (
    green_lagrange_strain,
    guccione_energy,
    holzapfel_ogden_energy,
    mooney_rivlin_energy,
    neo_hookean_energy,
)


def test_material_energies_vanish_at_identity() -> None:
    F = np.eye(3)
    assert abs(neo_hookean_energy(F, mu=2.0, kappa=100.0)) < 1e-12
    assert abs(mooney_rivlin_energy(F, c10=1.0, c01=0.5, kappa=100.0)) < 1e-12
    assert abs(guccione_energy(F, c=1.0, bf=8.0, bt=2.0, bfs=4.0)) < 1e-12
    assert abs(holzapfel_ogden_energy(F, a=0.1, b=5.0, af=1.0, bf=8.0, as_=0.5, bs=5.0, afs=0.2, bfs=5.0, kappa=100.0)) < 1e-12


def test_green_lagrange_extension_has_positive_axial_strain() -> None:
    F = np.diag([1.1, 1.0, 1.0])
    E = green_lagrange_strain(F)
    assert E[0, 0] > 0
    assert np.isclose(E[1, 1], 0.0)
