"""
    FedAvg
"""

from CNN import CNN


def fed_avg(models):
    total_count = 0
    for i in models:
        total_count += i['count']
    for i in models:
        i['weight'] = i['count'] / total_count

    # init new model state dict
    avg_model_sd = models[0]['model'].state_dict()
    avg_model_sd = {key: 0 for key in avg_model_sd}

    # average models
    for i in models:
        model_sd = i['model'].state_dict()
        model_weight = i['weight']
        for key in avg_model_sd:
            avg_model_sd[key] += model_sd[key] * model_weight

    new_model = CNN()
    new_model.load_state_dict(avg_model_sd)
    return new_model

def fed_nova(models,tau):
    return 0
