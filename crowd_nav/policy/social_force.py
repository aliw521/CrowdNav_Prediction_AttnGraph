import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class SOCIAL_FORCE(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'social_force'


    def predict(self, state):
        """
        Produce action for agent with circular specification of social force model.    用社会力量模型的循环规范为代理生成行动
        """
        # Pull force to goal
        delta_x = state.self_state.gx - state.self_state.px
        delta_y = state.self_state.gy - state.self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        desired_vx = (delta_x / dist_to_goal) * state.self_state.v_pref  # 得到期望的水平速度   
        #是为了将位移在距离上进行归一化，确保速度与目标位置的距离成比例。然后将归一化后的值乘以参考速度 state.self_state.v_pref，得到期望的水平速度。
        desired_vy = (delta_y / dist_to_goal) * state.self_state.v_pref # 得到期望的垂直速度
        KI = self.config.sf.KI # Inverse of relocation time K_i          #KI 表示重分配时间的倒数，用于控制当前速度与期望速度之间的差异
        curr_delta_vx = KI * (desired_vx - state.self_state.vx)         
        curr_delta_vy = KI * (desired_vy - state.self_state.vy)
        
        # Push force(s) from other agents
        A = self.config.sf.A # Other observations' interaction strength: 1.5            观测强度是1.5       A 是其他观测之间的相互作用强度，用于控制相互作用的强度。
        B = self.config.sf.B # Other observations' interaction range: 1.0               观测强度是1.0       B 是其他观测之间的相互作用范围，用于调整相互作用的距离
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            delta_x = state.self_state.px - other_human_state.px
            delta_y = state.self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)    
            interaction_vx += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_x / dist_to_human)
            interaction_vy += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_y / dist_to_human)
            #np.exp() 是 NumPy 库中的一个函数，它用于计算给定数值的指数函数值
            # (state.self_state.radius + other_human_state.radius - dist_to_human) 表示当前位置与其他人之间的距离差 距离差越小，指数越大
            #最后，将相互作用强度与相互作用方向在 x 轴和 y 轴上的分量相乘，并将结果累加到 interaction_vx 和 interaction_vy 中

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * self.config.env.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.config.env.time_step 
        过这样的计算，可以获得考虑了当前状态的自身速度变化和其他人的相互作用后的总体速度变化，在模拟或控制系统中有助于更新当前状态的速度信息。

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = state.self_state.vx + total_delta_vx
        new_vy = state.self_state.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])  用于计算给定向量的模长

        if act_norm > state.self_state.v_pref:
            return ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref)
        else:
            return ActionXY(new_vx, new_vy)
