import torch.nn as nn
import torch
import numpy as np

class GAN(nn.Module):
    def __init__(self,generator : nn.Module, discriminator : nn.Module, loss_func = nn.BCELoss()):
        """

        :param generator: the generator network used for generating images
        :param discriminator: the discriminator network used for discriminating between the two, if the discriminator
        is to receive extra information then this is given in the forward / training call. The output of the discriminator
        should be a single output between 0 and 1.
        :param loss_func: the instantiated loss function
        """
        super(GAN,self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss_func = loss_func


    def forward(self,x, extra=None):
        """

        :param x: input to the generator
        :param extra: any extra information that the discriminator should receive
        :return:
        """
        generated_image = self.generator.forward(x)
        disc_input = generated_image
        if extra is not None:
            # this is assuming dimension 1 is the channel dimension
            disc_input = torch.cat((disc_input,extra),dim=1)

        output = self.discriminator.forward(disc_input)
        return output


    def train(self, generator_xs, real_images, optimizer, extra_discriminator_xs=None, epochs=2000,verbose=True,
              lr=0.001, smoothing=1.0):
        """
        validation sets have been excluded as it's often best to validate by checking by eye the outputs of the network

        in our case the discriminator_xs will be the same as our input

        :param generator_xs: this is the noise or images that you feed into the generator network
        :param real_images: these are the set of images you are trying to recreate
        :param optimizer: the optimizer you want to use e.g. optimizer=torch.optim.Adam
        :param extra_discriminator_xs: these are extra inputs to the discriminator that will be concatenated with the
        input to the discriminator in our case you want to put in the corresponding down sampled image with the high res
        image
        :param verbose: not implemented yet
        :param lr: the learning rate for the discriminator and the generator
        :param smoothing: not implemented yet
        :return:
        """
        if extra_discriminator_xs is not None:
            if generator_xs.shape != extra_discriminator_xs.shape:
                raise ValueError("extra_discriminator_xs dimensions do no match those of generator_xs")

        # different learning rates may be needed for the discriminator and the generator
        self.d_optimizer = optimizer(self.discriminator.parameters(),lr)
        self.g_optimizer = optimizer(self.generator.parameters(),lr)

        # ones represent a real image
        # zeros represent a fake image

        #experiment with this as training may not work best like this and there may be huge jumps

        g_loss = self.update_generator(generator_xs, extra=extra_discriminator_xs)
        d_loss = self.update_discriminator(generator_xs, real_images, extra=extra_discriminator_xs)
        for i in range(0, epochs):
            if g_loss > d_loss:
                print("updating generator")
                g_loss = self.update_generator(generator_xs, extra=extra_discriminator_xs)
            else:
                print("updating discriminator")
                d_loss = self.update_discriminator(generator_xs, real_images, extra=extra_discriminator_xs)
            self._log(g_loss,d_loss,epochs,i)



    def _log(self,g_loss,d_loss,epochs,i):
        print("###############################")
        print("epoch number " + str(i) + " of " + str(epochs))
        print("Generator Loss: " + str(g_loss))
        print("Discriminator Loss: " + str(d_loss))
        print("###############################")

    def update_discriminator(self, gen_xs, real_imgs, optimizer=None, extra=None, label_smoothing=1.0):
        """
        If the optimizer is given it must be specifically for the discriminator network

        :param gen_xs:
        :param real_imgs:
        :param optimizer:
        :param extra:
        :return:
        """
        if optimizer is None:
            optimizer = self.d_optimizer


        gen_ys  = torch.zeros(gen_xs.shape[0], 1)
        real_ys = torch.ones(real_imgs.shape[0], 1)
        real_ys.fill_(label_smoothing)

        #the loss from fake images
        gen_outputs = self.forward(gen_xs, extra)
        gen_loss = self.loss_func(gen_outputs, gen_ys)


        #the loss from real images
        disc_input = torch.cat((real_imgs,extra),dim=1)
        real_outputs = self.discriminator.forward(disc_input)
        real_loss = self.loss_func(real_outputs, real_ys)


        self.discriminator.zero_grad()
        self.generator.zero_grad()

        loss = gen_loss + real_loss
        loss.backward()
        optimizer.step()

        return loss.detach().numpy().mean()



    def update_generator(self,xs, optimizer=None, extra=None):
        """

        :param xs: the input images
        :param optimizer: the optimizer to be used if None then the optimizer set in train is used
        :param extra: any extra data that should be fed into the discriminator
        :return: the loss of the generator
        """
        if optimizer is None:
            optimizer = self.g_optimizer

        # the generator will only ever aim for it's images to be recognized as real
        # the ys are a vector of ones equal in length to the batch_size
        ys = torch.ones(xs.shape[0], 1)



        outputs = self.forward(xs, extra)
        loss = self.loss_func(outputs, ys)
        self.discriminator.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().numpy().mean()


def test():
    generator = nn.Sequential(nn.Linear(5,5,bias=False),nn.Sigmoid())
    discriminator = nn.Sequential(nn.Conv1d(2,1,1), nn.Linear(5,1,bias=False), nn.Sigmoid())
    inputs = []
    inputs.append([[1,2,3,4,5]])
    inputs.append([[2,2,3,3,5]])
    inputs.append([[5,6,2,3,1]])
    inputs.append([[5, 6, 2, 3, 1]])
    expected = []
    expected.append([[5,4,3,2,1]])
    expected.append([[5, 4, 3, 4, 5]])
    expected.append([[5, 4, 3, 2, 5]])
    expected.append([[5, 4, 3, 0, 2]])
    inputs = np.array(inputs)
    expected = np.array(expected)
    inputs = torch.Tensor(inputs)
    expected = torch.Tensor(expected)
    # needs to be a seperate clone
    extras = inputs.clone().detach()
    inputs.requires_grad = True
    expected.requires_grad = True
    model = GAN(generator,discriminator)
    model.train(inputs,expected,torch.optim.Adam,extras)

test()
