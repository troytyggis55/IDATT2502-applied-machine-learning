from multithreaded_train import train_model

learning_rates = ("learning_rate", (1e-4, 5e-4, 7e-4, 1e-3), 3e-4)
steps_per_updates = ("steps_per_update", (10_000, 50_000, 75_000, 100_000), 20_000)
random_opponents = ("random_opponent_every_n_updates", (2, 10), 5)
n_steps = ("n_steps", (256, 512, 1024, 4096), 2048)
batch_sizes = ("batch_size", (32, 128, 256, 512), 64)
n_epochs = ("n_epochs", (2, 4, 6, 8), 10)
gammas = ("gamma", (0.92, 0.96, 0.98, 0.995), 0.99) # 12.5, 25, 50, 100, 200 steps ahead
gae_lambdas = ("gae_lambda", (0.85, 0.9, 0.98, 0.99), 0.95)
clip_ranges = ("clip_range", (0.1, 0.3, 0.4, 0.5), 0.2)
ent_coefs = ("ent_coef", (0.01, 0.02, 0.03, 0.04), 0.0)
vf_coefs = ("vf_coef", (0.25, 0.75, 1.0), 0.5)
max_grad_norms = ("max_grad_norm", (0.25, 0.75, 1.0), 0.5)
use_sdes = ("use_sde", (True), False)

hyperparameters = [steps_per_updates, random_opponents, n_steps, batch_sizes, n_epochs, gammas, gae_lambdas, clip_ranges, ent_coefs, vf_coefs, max_grad_norms, use_sdes]

if __name__ == "__main__":
    #train_model(name="models/benchmark", total_trainingsteps=3_000_000)
    for parameter in hyperparameters:
        for value in parameter[1]:
            name = f"models/{parameter[0]}/{value}"
            train_model(name=name, **{parameter[0]: value}, total_trainingsteps=3_000_000)

