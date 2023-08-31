policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.srnn import SRNN, selfAttn_merge_SRNN

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['srnn'] = SRNN
policy_factory['selfAttn_merge_srnn'] = selfAttn_merge_SRNN      #这段代码可用于实现策略的选择和配置，使得用户能够在运行时选择不同的导航策略，并根据策略名称使用相应的对象进行导航
