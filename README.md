# Shared Vehicle Rebalancing Operator

#### Designed, implemented, and tested by [Thien-An Bui](https://www.linkedin.com/in/thien-an-bui/)

## Overview
This project seeks to reduce instances of bike unavailability at stations where customer demand exists for shared vehicles. 
At a high level, we want to reward the agent when rides are taken and penalize it when there is demand for a ride and there is no vehicle available. 
One method looks to penalize the structure for empty vehicle stations (no vehicles = no rides can be taken) to ensure that stations have at least one (or a certain percentage threshold of) vehicle(s) for riders to use. 

Using reinforcement learning techniques, namely DQN and Q-Learning methods, we simulate bike stock in a docking station. We employ the rebalancing agent in two environments, testing both linear and nonlinear delta rates, to test how its adaptation abilities.

## Background
Consider the Divvy or Citi Bike stations in New York City, Chicago, etc. Shared vehicle stations can become overstocked and require rebalancing intervention to shift vehicle supply to understocked locations. 
Left alone, this issue can result in lost revenue opportunities, decreased customer retention rates, and misplaced inventory.

## Methodology and Setup
An agent needs an environment to act upon and provide it with states of the world, transition probabilities, etc. Likewise, it must have outlined rewards and penalties to help guide it towards a desired behavior.
In this project, we create the foundational components to shape our agent's decisions and enact them. 

##### Goal: The agent should strive to minimize intervention while keeping the station's inventory within the desired range.


![Setup](/Snapshots/Setup.PNG "Initial Setup Details")


### Reward Structure
We establish an incentive schema aimed at guiding our agent towards an efficient supply allocation strategy. At a high level, it contains the following components:
- A final reward or penalty at the end of each day.
- Heavy penalty thresholds for allowing the station to overfill or completely deplete in inventory.
- Moderate penalty thresholds directly outside the target range.
- No penalty zones marking the target range.
- A "fuel" cost for each unit moved.

For a detailed breakdown, see the image below.

![Reward Matrix](/Snapshots/Reward_layout.PNG "Reward Matrix")

The reward structure throughout each hour interval in a day can be seen below.


![Reward Visual](/Snapshots/Reward_visual.PNG "Parametrized Reward Structure")


### Simulated Environments
We employ the operator agent in two environments. The first (linear delta) environment tests whether the operator's predictive capacities and reward incentive structure are functioning as expected.
The second (nonlinear delta) adds randomness into the future state and introduces the ability for bike stock to decrease in the following time interval.

#### Linear Environment
![Linear Setup](/Snapshots/Linear_setup.PNG "Linear Environment")

#### Nonlinear Environment 
![Nonlinear Setup](/Snapshots/Nonlinear_setup.PNG "Nonlinear Environment")
