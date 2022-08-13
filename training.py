import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import numpy as np

def ganTrain(dataloader, discriminator, generator, device, epochs=20):
    print("Training")
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(discriminator.parameters(),
                                          lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(),
                                      lr=0.0001, betas=(0.5, 0.999))
    save_points = np.int32(np.array([0.5, 1]) * (epochs-1))
    generator_states = {}
    discriminator_states = {}
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            images = data[0].to(device).float()
            targets = data[1].to(device)

            y_real = torch.ones(images.shape[0], device=device)
            y_fake = torch.zeros(images.shape[0], device=device)

            discriminator.zero_grad()
            # Collapse the column dimension
            output_real = discriminator(images).view(-1)
            discriminator_real_loss = criterion(output_real, y_real)

            # Classify fake images
            noise = torch.randn(images.shape[0], generator.latent, device=device)
            fake_images = generator(noise)
            # Detach fake images from the graph
            # Otherwise backward() will clear all the variables which are needed
            # for updating the generator
            # Max 1 - log(D(G(z))
            output_fake = discriminator(fake_images.detach()).view(-1)
            discriminator_fake_loss = criterion(output_fake, y_fake)
            discriminator_loss = discriminator_real_loss + discriminator_fake_loss
            discriminator_loss.backward()
            optimizerD.step()

            # Train Generator
            # One term in the loss function
            generator.zero_grad()
            output_fake2 = discriminator(fake_images).view(-1)
            # Max log(D(G(z))
            generator_loss = criterion(output_fake2, y_real)
            generator_loss.backward()

            optimizerG.step()

        generator_states[epoch] \
            = generator.state_dict()

        discriminator_states[epoch] \
            = discriminator.state_dict()

        print(epoch)
        print(f"Discriminator loss is {discriminator_loss}")
        print(f"Generator loss is {generator_loss} \n")


    return generator_states, discriminator_states

def wganTrain(dataloader, discriminator, generator, device, epochs=20):
    print("Training")
    optimizerD = optim.RMSprop(discriminator.parameters(),
                            lr=0.00005)
    optimizerG = optim.RMSprop(generator.parameters(),
                            lr=0.00005)
    save_points = np.int32(np.array([0.5, 1]) * (epochs - 1))
    generator_states = {}
    discriminator_states = {}
    clip_value = 0.009
    n_critic = 5
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            images = data[0].to(device).float()
            # Create fake images
            optimizerD.zero_grad()
            noise = torch.randn(images.shape[0], generator.latent, device=device)
            fake_images = generator(noise)
            discriminator_loss = -torch.mean(discriminator(images)) + torch.mean(discriminator(fake_images))

            discriminator_loss.backward()
            optimizerD.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizerG.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(noise)
                # Adversarial loss
                generator_loss = -torch.mean(discriminator(gen_imgs))

                generator_loss.backward()
                optimizerG.step()

        generator_states[epoch] \
            = generator.state_dict()

        discriminator_states[epoch] \
            = discriminator.state_dict()

        print(epoch)
        print(f"Discriminator loss is {discriminator_loss.item()}")
        print(f"Generator loss is {generator_loss.item()} \n")

    return generator_states, discriminator_states

def wganGPTrain(dataloader, discriminator, generator, device, epochs=20):
    print("Training")
    optimizerD = optim.Adam(discriminator.parameters(),
                            lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(),
                            lr=0.0001, betas=(0.5, 0.999))
    save_points = np.int32(np.array([0.5, 1]) * (epochs - 1))
    generator_states = {}
    discriminator_states = {}
    gradient_coeff = 0.1
    n_critic = 5
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            images = data[0].to(device).float()
            # Create fake images
            optimizerD.zero_grad()

            noise = torch.randn(images.shape[0], generator.latent, device=device)
            fake_images = generator(noise)
            pred_real = discriminator(images)
            pred_fake = discriminator(fake_images.detach())
            # Grad pen
            alpha = torch.rand(images.size(0), 1).to(device)
            mixture_sample = alpha * images + (1 - alpha) * fake_images
            mixture_sample = torch.autograd.Variable(mixture_sample.to(device), requires_grad=True)
            output_mixture = discriminator(mixture_sample)
            grad_pen = gradientPenalty(mixture_sample, output_mixture, device)
            # WGAN Loss
            discriminator_loss = (torch.mean(pred_fake) - torch.mean(pred_real)) + grad_pen*gradient_coeff

            discriminator_loss.backward()
            optimizerD.step()

            # Train the generator every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizerG.zero_grad()
                # Generate a batch of images
                gen_imgs = generator(noise)
                # Adversarial loss
                generator_loss = -torch.mean(discriminator(gen_imgs))
                optimizerG.zero_grad()
                generator_loss.backward()
                optimizerG.step()

        generator_states[epoch] \
            = generator.state_dict()

        discriminator_states[epoch] \
            = discriminator.state_dict()

        print(epoch)
        print(f"Discriminator loss is {discriminator_loss.item()}")
        print(f"Generator loss is {generator_loss.item()} \n")

    return generator_states, discriminator_states

def gradientPenalty(mixture_samples, mixture_output, device):
    weight = torch.ones(mixture_output.size()).to(device)
    gradients = torch.autograd.grad(
     outputs=mixture_output,
     inputs=mixture_samples,
     grad_outputs=weight,
     create_graph=True,
     retain_graph=True,
     only_inputs=True)[0]


    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def trainCNN(trainloader, cnn, optimizer, device, epochs=20):
    loss_function = nn.CrossEntropyLoss()
    cnn_states = {}
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        cnn_states[epoch] \
            = cnn.state_dict()

    return cnn_states