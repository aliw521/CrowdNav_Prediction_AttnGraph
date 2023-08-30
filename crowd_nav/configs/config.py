import numpy as np
from arguments import get_args

class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # for now, import all args from arguments.py
    args = get_args()

    training = BaseConfig()
    training.device = "cuda:0" if args.cuda else "cpu"

    # general configs for OpenAI gym env
    env = BaseConfig()
    env.time_limit = 50
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    # if randomize human behaviors, set to True, else set to False
    env.randomize_attributes = True
    env.num_processes = args.num_processes
    # record robot states and actions an episode for system identification in sim2real
    env.record = False
    env.load_act = False

    # config for reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 0.25
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99

    # config for simulation
    sim = BaseConfig()
    sim.circle_radius = 6 * np.sqrt(2)   #这段代码设置了模拟器的配置，模拟器（sim）的圆形半径=6*根号2
    sim.arena_size = 6           # 场地尺寸是6
    sim.human_num = 20         #  人类数量是20
    # actual human num in each timestep, in [human_num-human_num_range, human_num+human_num_range]
    sim.human_num_range = 0
    sim.predict_steps = 5
    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    sim.predict_method = 'inferred'
    # render the simulation during training or not
    sim.render = False           #将模拟器的可视化渲染设置为false，不会形成图形界面或绘制结果。

    # for save_traj only
    render_traj = False       #不渲染轨迹，不会显示或者绘制物体的轨迹
    save_slides = False      #不会将模拟结果保存为幻灯片等格式
    save_path = None      #不会将模拟结果保存在指定的路径中

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method == 'inferred':
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # human config
    humans = BaseConfig()
    humans.visible = True
    # orca or social_force for now
    humans.policy = "orca"
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = "coordinates"   #人类所设置的传感器设置为坐标
    # FOV = this values * PI       #人类的视野设置为给定参数的两倍，表示360°
    humans.FOV = 2.                #fov的值在内部乘以Π，因此人类视野是2Π，模拟中的人类代理将能够360度全方位感知周围环境，使他们能够拥有完整的态势感知并根据环境的全面视图做出决策

    # a human may change its goal before it reaches its old goal
    # if randomize human behaviors, set to True, else set to False
    humans.random_goal_changing = True
    humans.goal_change_chance = 0.5

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step      
    humans.random_unobservability = False     #此参数设置为 False，表示人类代理的可观察性不是随机的。通过将其设置为 True，可以随机确定代理的可观察性。
    humans.unobservable_chance = 0.3    # 该参数设置为0.3，表示人类代理有30%的机会变得不可观察。不可观察性意味着代理无法被模拟中的其他代理或传感器感知或检测到。
    humans.random_policy_changing = False

    # robot config
    robot = BaseConfig()
    # whether robot is visible to humans (whether humans respond to the robot's motion)
    robot.visible = False
                                              # For baseline: srnn; our method: selfAttn_merge_srnn
    robot.policy = 'selfAttn_merge_srnn'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2
    # radius of perception range
    robot.sensor_range = 5

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle     整体工程学或者单轮车
    action_space.kinematics = "holonomic"        #整体运动学空间动作学=整体运动学

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10        #该参数设置为10，表示ORCA（最优相互碰撞避免）算法中邻居之间期望的距离。它确定每个智能体将其他智能体视为潜在碰撞对象的程度。
    orca.safety_space = 0.15       # time_horizon：该参数设置为5，表示ORCA算法中用于碰撞避免计算的时间范围。它指定代理预测未来多远的潜在碰撞并相应地调整其自身的运动
    orca.time_horizon = 5
    orca.time_horizon_obst = 5  #该参数设置为5，指定ORCA算法中专门用于考虑障碍物的时间范围。它决定了智能体预测与静态或移动障碍物的碰撞的未来距离。

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # config for data collection for training the GST predictor
    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    data.num_processes = 5
    data.data_save_dir = 'gst_updated/datasets/orca_20humans_no_rand'
    # number of seconds between each position in traj pred model
    data.pred_timestep = 0.25

    # config for the GST predictor
    pred = BaseConfig()
    # see 'gst_updated/results/README.md' for how to set this variable
    # If randomized humans: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
    # else: gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj
    pred.model_dir = 'gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj'

    # LIDAR config
    lidar = BaseConfig()
    # angular resolution (offset angle between neighboring rays) in degrees   # 角度分辨率（相邻射线之间的偏移角度），单位为度
    lidar.angular_res = 5
    # range in meters
    lidar.range = 10

    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = True
    sim2real.record = False
    sim2real.load_act = False      #表示禁用任何预先存在的操作数据的加载。此设置表明将在模拟或训练过程中生成或获取新的动作数据。
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True

    if sim.predict_method == 'inferred' and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != 'inferred' and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")
