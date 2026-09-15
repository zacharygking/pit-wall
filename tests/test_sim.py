import pytest

from pit_wall.calibrate import compare
from pit_wall.race import neutralised_laps
from pit_wall.sim import Strategy, simulate, actual_strategies


def test_replay_of_generating_model_reproduces_the_order(synthetic):
    model, strategies, race = synthetic
    sim = simulate(race, model, strategies, overtake_threshold=0.0)
    assert sim.order == race.classified_order()
    c = compare(race, sim)
    assert c["kendall_tau"] == pytest.approx(1.0)
    assert c["exact_positions"] == len(model.drivers)


def test_pit_stop_costs_pit_loss_and_resets_tyre_age(synthetic):
    model, _, race = synthetic
    d = model.drivers[0]
    only = {d: Strategy.from_list([(10, "HARD")])}
    sim = simulate(race, model, only, overtake_threshold=0.0)
    none = simulate(race, model, {}, overtake_threshold=0.0)
    # the in-lap and out-lap together cost the pit loss, less the fresh tyre's gain on the out-lap
    two_laps = sum(sim.lap_times[d][9:11]) - sum(none.lap_times[d][9:11])
    assert abs(two_laps - model.pit_loss) < 1.5
    # from then on the fresh tyre is quicker than the old one
    assert sim.lap_times[d][12] < none.lap_times[d][12]


def test_slower_car_cannot_pass_without_pace_advantage(synthetic):
    model, _, race = synthetic
    fast, slow = model.drivers[0], model.drivers[1]
    # put the slow car 0.2 s ahead after lap 1 and give the fast car only a tiny pace edge
    race.laps[(fast, 1)] = race.laps[(fast, 1)].__class__(fast, 1, race.laps[(slow, 1)].duration + 0.2, False, None)
    model.base[fast] = model.base[slow] - 0.1  # 0.1 s/lap quicker, under the 0.5 s threshold
    sim = simulate(race, model, {}, overtake_threshold=0.5, min_gap=0.7)
    assert sim.order.index(slow) < sim.order.index(fast)
    sim2 = simulate(race, model, {}, overtake_threshold=0.05, min_gap=0.7)
    assert sim2.order.index(fast) < sim2.order.index(slow)


def test_safety_car_bunches_the_field(synthetic):
    model, strategies, race = synthetic
    race.sc_laps = {20, 21, 22}
    sim = simulate(race, model, strategies)
    cum = {d: sim.cumulative[d][21] for d in model.drivers}  # after lap 22
    spread = max(cum.values()) - min(cum.values())
    assert spread < 5.0


def test_neutralised_laps_parses_windows():
    msgs = [
        {"date": "t1", "lap_number": 5, "message": "SAFETY CAR DEPLOYED"},
        {"date": "t2", "lap_number": 8, "message": "SAFETY CAR IN THIS LAP"},
        {"date": "t3", "lap_number": 20, "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": "t4", "lap_number": 21, "message": "VIRTUAL SAFETY CAR ENDING"},
        {"date": "t5", "lap_number": 30, "message": "RED FLAG"},
    ]
    sc, vsc, red = neutralised_laps(msgs, 40)
    assert sc == {5, 6, 7, 8}
    assert vsc == {20, 21}
    assert red == {30}


def test_actual_strategies_come_from_stints(synthetic):
    _, strategies, race = synthetic
    assert actual_strategies(race) == strategies
