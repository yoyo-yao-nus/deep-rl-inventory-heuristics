# same for hurl agents
hyperparameters = {
    'dqn': {
        'lr': [1e-5, 1e-4, 1e-3]
    },
    'a2c': {
        'lr_a': [1e-5, 1e-4, 1e-3],
        'lr_c': [1e-5, 1e-4, 1e-3]
    },
    'ppo': {
        'lr_a': [5e-5 ,1e-4, 1e-3],
        'lr_c': [5e-5 ,1e-4, 1e-3],
        'lmbda': [0.95]
    },
    'sac': {
        'lr_a': [5e-5 ,1e-4, 1e-3],
        'lr_c': [5e-5 ,1e-4, 1e-3],
        'lr_alpha': [1e-3],
        'target_entropy': [-1],
        'tau': [0.005]
    }
}
# env parameters: (p,l)
param_grid = [(4, 2),\
        (4, 3),\
        (4, 4),\
        (9, 2),\
        (9, 3),\
        (9, 4)]