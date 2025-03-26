# Task Description

> "You take the blue pill... the story ends, you wake up in your bed and believe whatever you want to believe. You take the red pill... you stay in Wonderland, and I show you how deep the rabbit hole goes." ———— *The Matrix*

There are several capsules scattered on the table. Collect ONLY the blue ones into the bottle on the table.

# Custom Objects

Bottle - [A four-wall bin](https://github.com/haosulab/ManiSkill/issues/964#issuecomment-2753419887).

Capsules. 

Constraints: capsules cannot be spawned: 
* inside the bin
* outside the table

# Reward Function Design

* distances of capsules from the center
* the number of capsules in the bin
* if ONLY blue capsules are collected, give full rewards


# Extendable Aspects

* Add a slidable lid for the bin.
    * Maybe try using [articulations](https://sapien-sim.github.io/docs/user_guide/getting_started/create_articulations.html) (joints & links)?
* Different robots?
* More capsules and more colors?
* More realistic scene?
* Different bins?
