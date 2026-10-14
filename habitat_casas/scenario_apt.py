"""Places, sensors and a fixed two-resident morning in the ReplicaCAD apartment (v3_sc3_staging_16).

Coordinates come from the scene's furniture (frl_apartment_*) and articulated
objects. Layout: kitchen along the x = -2.3 wall (counter, cupboards, fridge
at the z = 3.6 end), sofa in the middle, dining table + beanbag + TV around
z = 6-7, desk (table_01) near the door, door2 at z = -2.65 leading to the
back area that stands in for "outside".
"""
from habitat_casas.activities import (
    Activity, Close, Home, Open, Pick, Place, Routine, Wait, Walk,
)

HOME = Home(
    places={
        # look-at points (x, y, z)
        "outside": (-1.2, 0.5, -3.8),
        "door": (-0.35, 1.0, -2.65),
        "entry": (-0.6, 0.5, -1.9),
        "desk": (-2.05, 0.66, -1.17),
        "desk_2": (-2.05, 0.66, -0.6),
        "dresser": (0.09, 1.03, 0.43),
        "counter_wash": (-2.3, 0.95, 1.0),
        "counter_cook": (-2.3, 0.95, 2.1),
        "fridge": (-1.86, 1.0, 3.99),
        "sofa": (1.51, 0.4, 2.81),
        "sofa_2": (2.4, 0.4, 2.81),
        "wardrobe": (0.46, 1.4, 4.63),
        "medicine_cabinet": (3.7, 0.5, 4.8),
        "dining_table": (0.81, 0.8, 6.0),
        "dining_table_2": (0.81, 0.8, 5.6),
        "tv": (0.41, 1.5, 6.09),
        "tv_2": (0.41, 1.5, 6.09),
        # surfaces to put things on
        "counter_top": (-2.25, 0.98, 2.1),
        "counter_top_2": (-2.25, 0.98, 1.25),
        "dining_table_top": (0.75, 1.0, 5.75),
        "dining_table_top_2": (1.0, 1.0, 5.65),
    },
    # second spots exist so two residents never stand in the same place.
    # Navmesh snapping would put "tv" on the beanbag; tv_2 is 0.8 m from tv.
    stand={"tv": (2.0, 0.11, 6.6), "tv_2": (2.8, 0.11, 6.4)},
)

FOOD = "018_plum_:0000"  # starts inside the fridge (top compartment)

ROUTINES = [
    Routine(agent=0, start_place="outside", activities=[
        Activity("Enter_Home", [
            Walk("door"), Open("door2", "body"), Walk("entry"), Close("door2", "body"),
        ]),
        Activity("Meal_Preparation", [
            Walk("fridge"), Open("fridge", "top_door"), Pick(FOOD), Close("fridge", "top_door"),
            Walk("counter_cook"), Open("kitchen_counter", "drawer1_top"), Wait(5),
            Close("kitchen_counter", "drawer1_top"), Place(FOOD, "counter_top"), Wait(180),
        ]),
        Activity("Eat", [
            Pick(FOOD), Walk("dining_table"), Place(FOOD, "dining_table_top"), Wait(300),
        ]),
        Activity("Wash_Dishes", [Walk("counter_wash"), Wait(120)]),
        Activity("Watch_TV", [Walk("tv"), Wait(600)]),
    ]),
    Routine(agent=1, start_place="sofa", activities=[
        Activity("Relax", [Wait(60)]),
        Activity("Take_Medicine", [
            Walk("medicine_cabinet"), Open("cabinet", "left_door"), Wait(15),
            Close("cabinet", "left_door"),
        ]),
        Activity("Snack", [
            Walk("fridge"), Open("fridge", "bottom_door"), Wait(10),
            Close("fridge", "bottom_door"), Walk("sofa"), Wait(240),
        ]),
        Activity("Work", [Walk("desk"), Wait(480)]),
        Activity("Get_Dressed", [
            Walk("dresser"), Open("chestOfDrawers", "drawer_topL"), Wait(30),
            Close("chestOfDrawers", "drawer_topL"), Walk("wardrobe"), Wait(60),
        ]),
        Activity("Leave_Home", [
            Walk("door"), Open("door2", "body"), Walk("outside"), Close("door2", "body"),
        ]),
    ]),
]


# ------------------------------------------------------------------ sensors
from habitat_casas.sensors import Contact, Item, MotionZone  # noqa: E402

MOTION = [
    # ceiling PIRs, floor rectangles x=(min, max), z=(min, max)
    MotionZone("M001", "entry", (-1.5, 0.5), (-2.6, -1.3)),
    MotionZone("M002", "desk", (-2.7, -1.0), (-2.0, -0.3)),
    # neighbouring spot sensors overlap by ~0.4 m, as real PIR cones do
    MotionZone("M003", "kitchen_sink", (-2.0, -0.8), (0.3, 1.8)),
    MotionZone("M004", "kitchen_stove", (-2.0, -0.8), (1.4, 3.1)),
    MotionZone("M005", "kitchen_fridge", (-2.0, -0.8), (2.7, 4.2)),
    MotionZone("M006", "living_dresser", (-0.8, 1.0), (-0.5, 1.8)),
    MotionZone("M007", "sofa", (1.0, 3.2), (1.5, 3.5)),
    MotionZone("M008", "hallway", (-0.8, 1.0), (1.8, 4.9)),
    MotionZone("M009", "dining", (0.8, 1.6), (5.0, 6.4)),
    MotionZone("M010", "tv_corner", (1.6, 3.0), (6.2, 7.6)),
    MotionZone("M011", "cabinet", (3.0, 4.5), (3.5, 6.0)),
    MotionZone("M012", "east_room", (2.0, 4.5), (-1.5, 1.5)),
    # room-wide "area" sensors on top of the spot sensors (like CASAS area PIRs)
    MotionZone("M013", "kitchen_area", (-2.0, -0.6), (0.3, 4.2)),
    MotionZone("M014", "living_area", (-0.8, 3.2), (-0.5, 3.5)),
    MotionZone("M015", "dining_tv_area", (0.8, 3.0), (5.0, 7.6)),
]

CONTACTS = [
    Contact("D001", "door2", "body"),  # front door
    Contact("D002", "fridge", "top_door"),
    Contact("D003", "fridge", "bottom_door"),
    Contact("D004", "kitchen_counter", "drawer1_top"),
    Contact("D005", "cabinet", "left_door"),  # medicine cabinet
    Contact("D006", "chestOfDrawers", "drawer_topL"),
    Contact("D007", "kitchenCupboard_01", "kitchencupboard_doorWhole_1L"),
    Contact("D008", "kitchen_counter", "drawer2_top"),
    Contact("D009", "kitchen_counter", "drawer2_middle"),
]

ITEMS = [
    Item("I001", FOOD),  # plum on its fridge shelf
    Item("I002", "024_bowl_:0001"),  # bowl in the fridge
    Item("I003", "002_master_chef_can_:0000"),  # coffee can on the counter
    Item("I004", "012_strawberry_:0000"),  # fruit on the counter
    Item("I005", "024_bowl_:0000"),  # bowl on the counter
]
