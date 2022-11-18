"""
    FedAvg
"""

from CNN import CNN


def fed_avg(models):
    n = len(models)
    avg_model = models[0]
    sd_avg_model = avg_model.state_dict()

    for i in range(1, n):
        sd_model = models[i].state_dict()
        for key in sd_model:
            sd_avg_model[key] = sd_avg_model[key] + sd_model[key]

    for key in sd_avg_model:
        sd_avg_model[key] = sd_avg_model[key] / n

    new_model = CNN()
    new_model.load_state_dict(sd_avg_model)
    return new_model
