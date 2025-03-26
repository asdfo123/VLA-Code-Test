# Task Description

Several capsules are scattered on the table. Help me collect them into the bottle on the table.

# Custom Objects

[A four-wall bin](https://github.com/haosulab/ManiSkill/issues/964#issuecomment-2753419887).

Capsules. 

Constraints: capsules cannot be spawned: 
* inside the bin
* outside the table

# Reward Function Design

* distances of capsules from the center
* the number of capsules in the bin
* if all three capsules are collected, give a full reward


# Extendable Aspects

* Add a slidable lid for the bin.
    * Maybe try using [articulations](https://sapien-sim.github.io/docs/user_guide/getting_started/create_articulations.html) (joints & links)?
* Different robots?
* Customized number of capsules? Or not.
* More realistic scene?