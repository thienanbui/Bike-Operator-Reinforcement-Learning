# Shared Vehicle Rebalancing Operator

#### Designed, implemented, and tested by [Thien-An Bui](https://www.linkedin.com/in/thien-an-bui/)

## Overview
This project seeks to reduce instances of bike unavailability at stations where customer demand exists for shared vehicles. 
At a high level, we want to reward the agent when rides are taken and penalize it when there is demand for a ride and there is no vehicle available. 
One method looks to penalize the structure for empty vehicle stations (no vehicles = no rides can be taken) to ensure that stations have at least one (or a certain percentage threshold of) vehicle(s) for riders to use. 

We employ reinforcement learning techniques, namely DQN and Q-Learning methods, to simulate bike stock in a docking station. We employ the rebalancing agent in two environments, testing both linear and nonlinear delta rates, to test how its adaptation abilities.

## Background
Consider the Divvy or Citi Bike stations in New York City, Chicago, etc. Shared vehicle stations can become overstocked and require rebalancing intervention to shift vehicle supply to understocked locations. 
Left alone, this issue can result in lost revenue opportunities, decreased customer retention rates, and misplaced inventory.

## Simualated Environments
We employ the operator agent in two environments. The first (linear delta) environment tests whether the operator's predictive capacities and reward incentive structure are functioning as expected.
The second (nonlinear delta) adds randomness into the future state and introduces the ability for bike stock to decrease in the following time interval.

### Linear Environment
### NonLinear Environment 
