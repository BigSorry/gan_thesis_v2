import torch

def _getSamples(generator, device, samples):
    with torch.no_grad():
        noise = torch.randn(samples, generator.latent, device=device)
        generated = generator(noise).detach().cpu().numpy()

    return generated

def getGeneratedSamples(states_info, generator, device, samples):
    generated_sample_info = {}
    for epoch, model_state in states_info.items():
            generator.load_state_dict(model_state)
            generated_samples = _getSamples(generator, device, samples)
            generated_sample_info[epoch] = generated_samples

    return generated_sample_info