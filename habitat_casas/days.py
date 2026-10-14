"""Random two-resident days in the ReplicaCAD apartment.

generate_day(seed) returns one Routine per resident. Each day mixes:
  - individual activities in random order with random durations,
  - joint activities that both residents do at the same time in the same
    rooms (cooking together, eating together, watching TV / sitting on the
    sofa together), aligned with Sync steps,
  - sometimes one resident going out and coming back (resident count changes).

Residents keep their own seats (R1: desk/sofa/tv/..., R2: the "_2" spots) so
the two humanoids never stand in the same place during joint activities.
"""
import random
from typing import List, Tuple

from habitat_casas.activities import (
    Activity, Close, Open, Pick, Place, Routine, Sync, Wait, Walk,
)

LABELS = [
    "Enter_Home", "Leave_Home", "Away", "Meal_Preparation", "Eat", "Wash_Dishes",
    "Snack", "Take_Medicine", "Get_Dressed", "Work", "Relax", "Watch_TV",
]

FRIDGE_FOOD = ["018_plum_:0000", "024_bowl_:0001"]  # behind the fridge top door
COUNTER_FOOD = ["012_strawberry_:0000", "024_bowl_:0000", "002_master_chef_can_:0000"]
# where to stand to reach each counter item (the can is by the stove, the
# others by the sink; the far spot is 1.4-1.6 m away)
COUNTER_SPOT = {"012_strawberry_:0000": "counter_wash", "024_bowl_:0000": "counter_wash",
                "002_master_chef_can_:0000": "counter_cook"}


def _seat(base: str, r: int) -> str:
    return base if r == 0 else f"{base}_2"


class _Day:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.fridge = list(FRIDGE_FOOD)
        self.counter = list(COUNTER_FOOD)
        self.rng.shuffle(self.fridge)
        self.rng.shuffle(self.counter)
        self.food = {0: None, 1: None}  # (object, stand spot to pick it up from)

    def mins(self, lo: float, hi: float) -> float:
        return 60 * self.rng.uniform(lo, hi)

    # ---- individual activities
    def take_medicine(self, r):
        return Activity("Take_Medicine", [
            Walk("medicine_cabinet"), Open("cabinet", "left_door"),
            Wait(self.rng.uniform(8, 25)), Close("cabinet", "left_door")])

    def get_dressed(self, r):
        return Activity("Get_Dressed", [
            Walk("dresser"), Open("chestOfDrawers", "drawer_topL"), Wait(self.rng.uniform(15, 45)),
            Close("chestOfDrawers", "drawer_topL"), Walk("wardrobe"), Wait(self.mins(0.5, 2))])

    def work(self, r):
        return Activity("Work", [Walk(_seat("desk", r)), Wait(self.mins(5, 15))])

    def relax(self, r):
        return Activity("Relax", [Walk(_seat("sofa", r)), Wait(self.mins(3, 10))])

    def watch_tv(self, r):
        return Activity("Watch_TV", [Walk(_seat("tv", r)), Wait(self.mins(5, 15))])

    def snack(self, r):
        steps = [Walk("fridge"), Open("fridge", "bottom_door"), Wait(self.rng.uniform(5, 15)),
                 Close("fridge", "bottom_door")]
        if self.counter and self.rng.random() < 0.5:
            obj = self.counter.pop()  # grab fruit, eat it at the table
            steps += [Walk(COUNTER_SPOT[obj]), Pick(obj),
                      Walk(_seat("dining_table", r)), Place(obj, _seat("dining_table_top", r))]
        steps += [Walk(_seat("sofa", r)), Wait(self.mins(1, 4))]
        return Activity("Snack", steps)

    def wash_dishes(self, r):
        return Activity("Wash_Dishes", [
            Walk("counter_wash"), Open("kitchen_counter", "drawer2_top"), Wait(self.rng.uniform(5, 15)),
            Close("kitchen_counter", "drawer2_top"), Wait(self.mins(2, 6))])

    def go_out(self, r) -> List[Activity]:
        return [
            Activity("Leave_Home", [Walk("door"), Open("door2", "body"), Walk("outside"),
                                    Close("door2", "body")]),
            Activity("Away", [Wait(self.mins(5, 20))]),
            Activity("Enter_Home", [Walk("door"), Open("door2", "body"), Walk("entry"),
                                    Close("door2", "body")]),
        ]

    INDIVIDUAL = ["take_medicine", "get_dressed", "work", "relax", "snack", "watch_tv"]

    def individual_block(self, r, k_lo, k_hi, exclude=()):
        names = [n for n in self.INDIVIDUAL if n not in exclude]
        picks = self.rng.sample(names, self.rng.randint(k_lo, k_hi))
        return [getattr(self, n)(r) for n in picks]

    # ---- joint activities (same label for both residents, Sync-aligned)
    def cook(self, r, tag):
        """r=0 (or the solo cook) uses fridge + stove; the helper preps at the sink."""
        role_main = r == 0 or tag is None
        steps = [Sync(tag)] if tag else []
        if role_main:
            food = self.fridge.pop() if self.fridge else None
            steps += [Walk("fridge"), Open("fridge", "top_door")]
            if food:
                steps += [Pick(food)]
            steps += [Close("fridge", "top_door"), Walk("counter_cook"),
                      Open("kitchen_counter", "drawer1_top"), Wait(self.rng.uniform(3, 8)),
                      Close("kitchen_counter", "drawer1_top")]
            if food:
                steps += [Place(food, "counter_top")]
            spot = "counter_cook"
        else:
            food = self.counter.pop() if self.counter else None
            if food:
                steps += [Walk(COUNTER_SPOT[food]), Pick(food)]
            steps += [Walk("counter_wash")]
            if food:
                steps += [Place(food, "counter_top_2")]
            steps += [Open("kitchen_counter", "drawer2_middle"), Wait(self.rng.uniform(3, 8)),
                      Close("kitchen_counter", "drawer2_middle")]
            spot = "counter_wash"
        steps += [Wait(self.mins(3, 8))]
        self.food[r] = (food, spot) if food else None
        return Activity("Meal_Preparation", steps)

    def eat(self, r, tag):
        steps = [Sync(tag)] if tag else []
        food = None
        if self.food[r]:
            food, spot = self.food[r]
            steps += [Walk(spot), Pick(food)]
        steps += [Walk(_seat("dining_table", r))]
        if food:
            steps += [Place(food, _seat("dining_table_top", r))]
        steps += [Wait(self.mins(6, 12))]
        return Activity("Eat", steps)

    def together_on_couch(self, r, tag, kind):
        place = _seat("tv" if kind == "Watch_TV" else "sofa", r)
        return Activity(kind, [Sync(tag), Walk(place), Wait(self.mins(8, 20))])


def _merge(acts: List[Activity]) -> List[Activity]:
    """Back-to-back activities with the same label become one activity."""
    out: List[Activity] = []
    for a in acts:
        if out and out[-1].label == a.label:
            out[-1] = Activity(a.label, out[-1].steps + a.steps)
        else:
            out.append(a)
    return out


def generate_day(seed: int) -> Tuple[List[Routine], dict]:
    d = _Day(seed)
    rng = d.rng
    acts = {0: [], 1: []}
    info = {"seed": seed}

    # someone may arrive home at the start of the recording
    starts = {0: rng.choice(["sofa", "desk", "dresser"]),
              1: rng.choice(["sofa_2", "desk_2", "tv_2"])}
    late = rng.choice([0, 1]) if rng.random() < 0.3 else None
    if late is not None:
        starts[late] = "outside"
        acts[late] += [Activity("Enter_Home", [Walk("door"), Open("door2", "body"),
                                               Walk("entry"), Close("door2", "body")])]
    info["arrives_late"] = None if late is None else f"R{late + 1}"

    # 1) individual morning
    for r in (0, 1):
        acts[r] += d.individual_block(r, 1, 2)

    # 2) breakfast: together (cook + eat with Sync) or one resident alone
    if rng.random() < 0.8:
        info["breakfast"] = "joint"
        for r in (0, 1):
            acts[r] += [d.cook(r, "cook"), d.eat(r, "eat")]
        washer = rng.choice([0, 1])
        acts[washer] += [d.wash_dishes(washer)]
        acts[1 - washer] += [d.relax(1 - washer)]
    else:
        solo = rng.choice([0, 1])
        info["breakfast"] = f"solo_R{solo + 1}"
        acts[solo] += [d.cook(solo, None), d.eat(solo, None), d.wash_dishes(solo)]
        acts[1 - solo] += d.individual_block(1 - solo, 2, 3)

    # 3) midday: individual activities, maybe one resident goes out and back
    out = rng.choice([0, 1]) if rng.random() < 0.5 else None
    info["goes_out"] = None if out is None else f"R{out + 1}"
    for r in (0, 1):
        acts[r] += d.individual_block(r, 1, 2)
        if r == out:
            acts[r] += d.go_out(r)

    # 4) relaxing together
    if rng.random() < 0.75:
        kind = rng.choice(["Watch_TV", "Relax"])
        info["together"] = kind
        for r in (0, 1):
            acts[r] += [d.together_on_couch(r, "couch", kind)]
    else:
        info["together"] = None

    # 5) tail
    for r in (0, 1):
        acts[r] += d.individual_block(r, 0, 2, exclude=("watch_tv", "relax"))

    routines = [Routine(agent=r, start_place=starts[r], activities=_merge(acts[r])) for r in (0, 1)]
    return routines, info
