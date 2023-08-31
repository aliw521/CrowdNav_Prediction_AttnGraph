# Simulation Framework
## Environment
The environment contains n+1 agents. N of them are humans controlled by certain a fixed
policy. The other is robot and it's controlled by a trainable policy.
The environment is built on top of OpenAI gym library, and has implemented two abstract methods.
* reset(): the environment will reset positions for all the agents and return observation 
for robot. Observation for one agent is the observable states of all other agents.
* step(action): taking action of the robot as input, the environment computes observation
for each agent and call agent.act(observation) to get actions of agents. Then environment detects
whether there is a collision between agents. If not, the states of agents will be updated. Then 
observation, reward, done, info will be returned.


## Agent
Agent is a base class, and has two derived class of human and robot. Agent class holds
all the physical properties of an agent, including position, velocity, orientation, policy and etc.
* visibility: humans and robot can be set to be visible or invisible
* sensor: can be either visual input or coordinate input        可以是视觉输入或坐标输入
* kinematics: can be either holonomic (move in any direction) or unicycle (has rotation constraints)   可以是整体运动学（向任意方向移动），也可以是单轮运动学（有旋转限制）
* act(observation): transform observation to state and pass it to policy     将观察结果转换为状态并传递给策略


## Policy
Policy takes state as input and output an action. Current available policies:
* ORCA: model other agents as velocity obstacles to find optimal collision-free velocities under reciprocal assumption  将其他代理建模为速度障碍，以便在互惠假设下找到最佳的无碰撞速度
* Social force: models the interactions in crowds using attractive and  repulsive  forces
* DS-RNN: our method


## State
There are multiple definition of states in different cases. The state of an agent representing all
the knowledge of environments is defined as JointState, and it's different from the state of the whole environment.
* ObservableState: position, velocity, radius of one agent  半径
* FullState: position, velocity, radius, goal position, preferred velocity, rotation of one agent  首选速度和朝向角度
* JoinState: concatenation of one agent's full state and all other agents' observable states   将一个智能体的完整状态和其他智能体的可观测状态串联起来的状态


## Action
There are two types of actions depending on what kinematics constraint the agent has.
* ActionXY: (vx, vy) if kinematics == 'holonomic'
* ActionRot: (velocity, rotation angle) if kinematics == 'unicycle'
* 整体运动学和单论运动学是描述物体运动的两种不同视角。

整体运动学（holonomic kinematics）是指在描述物体运动时考虑了物体所有自由度的运动学。具体来说，整体运动学假设物体可以在三维空间中自由运动，并且可以以任意方向和速度进行移动。因此，在整体运动学中，物体的位置和速度可以在三个坐标轴上独立地表示和控制。例如，一个机器人在平面内的全向移动就属于整体运动学。

单论运动学（non-holonomic kinematics）则是指在描述物体运动时仅考虑了部分自由度的运动学。具体来说，单论运动学假设物体在其运动过程中存在运动约束，即物体不能以任意速度和方向移动。这些运动约束可以是限制物体的速度或者朝向角度的范围。因此，在单论运动学中，物体的位置和速度之间存在一定的限制关系。例如，一个车辆只能沿着预定路径行驶，不能实现全向移动，这时就需要使用单论运动学来描述车辆的运动。

总而言之，整体运动学适用于能够以任意方向和速度自由移动的物体，而单论运动学适用于由于运动约束而无法以任意速度和方向移动的物体
