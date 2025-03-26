import sapien
import random
from mani_skill.utils.building import articulations, actors

def random_color():
    """Generate a random RGBA color."""
    return [random.random(), random.random(), random.random(), 1.0]  # RGB with full alpha

def create_random_item(scene, index, cube_half_size, radius):
    """Create a random item with a random shape and color."""
    item_type = random.choice(["sphere", "box"])
    color = random_color()
    if item_type == "sphere":
        item = actors.build_sphere(
            scene,
            radius=random.uniform(radius, 3*radius),  # Random radius
            color=color,
            name=f"sphere_{index}",
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )
    elif item_type == "box":
        item = actors.build_box(
            scene,
            half_sizes= [random.uniform(cube_half_size, 2*cube_half_size), random.uniform(cube_half_size, 2*cube_half_size), random.uniform(cube_half_size, 2*cube_half_size)],  # Random half sizes
            color=color,
            name=f"box_{index}",
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )
    elif item_type == "cylinder":
        item = actors.build_cylinder(
            scene,
            radius=random.uniform(radius, 3*radius),  # Random radius
            half_length=random.uniform(2*radius, 5*radius),  # Random height
            color=color,
            name=f"cylinder_{index}",
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )
    elif item_type == "cube":
        item = actors.build_box(
            scene,
            half_sizes=[random.uniform(cube_half_size, 3*cube_half_size)] * 3,
            color=color,
            name=f"cube_{index}",
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    return item

def create_random_items(scene, n_items, cube_half_size=0.02, radius=0.02):
    """Create multiple random items."""
    items = []
    for i in range(n_items):
        items.append(create_random_item(scene, i, cube_half_size, radius))
    return items

# Example usage:
# items = create_random_items(self.scene, n_items)
