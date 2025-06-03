from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression

# import open bandit pipeline (obp)
import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_reward_function
)
from obp.policy import IPWLearner, Random
from obp.policy import BernoulliTS

from obp.ope import (
    OffPolicyEvaluation, 
    RegressionModel,
    InverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust
)

from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW

# (1) Data Loading and Preprocessing
dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
bandit_feedback = dataset.obtain_batch_bandit_feedback()

# (2) Production Policy Replication
evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True, # replicate the policy in the ZOZOTOWN production
    campaign="all",
    random_state=12345
)
action_dist = evaluation_policy.compute_batch_action_dist(
    n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
)
p = 0.5

value_lst = []
p = 0
for k in range(100):
    evaluation_policy = BernoulliTS(
    n_actions=dataset.n_actions,
    len_list=dataset.len_list,
    is_zozotown_prior=True, # replicate the policy in the ZOZOTOWN production
    campaign="all",
    random_state=12345
)
    action_dist = evaluation_policy.compute_batch_action_dist(
        n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
    )
    choice_arr = np.zeros((10000, 80,3))
    for j in range(3):
        for i in range(10000):
            a = action_dist[i, :, j]
            b = np.ones(80)/80
            rand = np.random.uniform()
            if rand <= p:
                choice_arr[i,:,j] = a
            else:
                choice_arr[i,:,j] = b

    # (3) Off-Policy Evaluation
    ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
    estimated_policy_value = ope.estimate_policy_values(action_dist=choice_arr)
    relative_policy_value_of_bernoulli_ts = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
    value_lst.append(relative_policy_value_of_bernoulli_ts)

# experiment with different p and select results

import random
 
def random_split(n,p):
    data = [i for i in range(n)]

    random.shuffle(data)

    split_point = int(len(data)*p)

    group1 = data[:split_point]
    group2 = data[split_point:]

    return group1, group2

p_lst = [1-0.1*x for x in range(1,10)]
mean_lst = []
std_lst = []
for p in p_lst:
    value_lst = []
    for k in range(5000):
        group1, group2 = random_split(10000,p)
        evaluation_policy = BernoulliTS(
        n_actions=dataset.n_actions,
        len_list=dataset.len_list,
        is_zozotown_prior=True, # replicate the policy in the ZOZOTOWN production
        campaign="all",
        random_state=12345
    )
        action_dist = evaluation_policy.compute_batch_action_dist(
            n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
        )
        choice_arr = np.zeros((10000, 80,3))
        for j in range(3):
            for i in range(10000):
                a = action_dist[i, :, j]
                b = np.ones(80)/80
                
                if i in group1:
                    choice_arr[i,:,j] = a
                else:
                    choice_arr[i,:,j] = b

        # (3) Off-Policy Evaluation
        ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
        estimated_policy_value = ope.estimate_policy_values(action_dist=choice_arr)
        relative_policy_value_of_bernoulli_ts = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
        value_lst.append(relative_policy_value_of_bernoulli_ts)
    mean_lst.append(np.mean(value_lst))
    print(np.mean(value_lst))
    std_lst.append(np.std(value_lst))
    print(np.std(value_lst))

# use quadratic function to approximate

import numpy as np

x = np.array([0.1*t for t in range(0,11)])
y = mean_lst

from sklearn.linear_model import LinearRegression

# Transform the data to include an x^2 term
x_b = np.c_[x, x**2]

# Create and fit the model
model = LinearRegression()
weights = 1.0 /  np.array([0.1+ t for t in range(0,11)])

model.fit(x_b, y, weights)

print(model.coef_)
# Coefficients
a = model.coef_[1]  # Coefficient for x^2
b = model.coef_[0]  # Coefficient for x
c = model.intercept_  # Intercept

# Display the results
print("Fitted quadratic function: y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(a, b, c))
