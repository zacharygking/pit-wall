import pytest

from pit_wall.model import clean_laps, fit


def test_fit_recovers_known_parameters(synthetic):
    model, strategies, race = synthetic
    fitted = fit(race)
    assert fitted.reference_compound in fitted.compounds
    # degradation slopes and fuel effect are recovered to a few thousandths of a second
    for c in ("SOFT", "MEDIUM", "HARD"):
        assert fitted.deg1[c] == pytest.approx(model.deg1[c], abs=3e-3)
    assert fitted.fuel == pytest.approx(model.fuel, abs=2e-3)
    # compound offsets are recovered relative to the reference compound
    ref = fitted.reference_compound
    for c in ("SOFT", "HARD"):
        want = model.offset[c] - model.offset[ref]
        assert fitted.offset[c] - fitted.offset[ref] == pytest.approx(want, abs=0.05)
    # base pace differences between drivers survive the reparameterisation
    d0, d1 = model.drivers[0], model.drivers[-1]
    assert (fitted.base[d1] - fitted.base[d0]) == pytest.approx(model.base[d1] - model.base[d0], abs=0.05)
    assert fitted.sigma < 0.05


def test_pit_loss_is_estimated_from_in_and_out_laps(synthetic):
    model, _, race = synthetic
    fitted = fit(race)
    assert fitted.pit_loss == pytest.approx(model.pit_loss, abs=0.3)


def test_clean_laps_exclude_lap1_inlaps_and_outlaps(synthetic):
    _, strategies, race = synthetic
    rows = clean_laps(race)
    laps = {(d, n) for d, n, *_ in rows}
    for d, strat in strategies.items():
        assert (d, 1) not in laps
        for pit_lap, _ in strat.pits:
            assert (d, pit_lap) not in laps
            assert (d, pit_lap + 1) not in laps
