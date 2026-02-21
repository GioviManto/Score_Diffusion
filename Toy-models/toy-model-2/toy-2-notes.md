Context. We are investigating the properties of the score function in a system that moves over time. We call this natural movement as "internal time". When measuing this system, we obtain data that incorporate this axis. For a single observation, we have multiple instances. For example, videos are such data type, as they have internal time and each video is made of a sequenceo of frames. We aim at interpratbility. Therefore, we simplify the problem to the bone and then use tools like algebra and animations. 

Previous toy model. The goal of the first toy model is to show how the forward and backward processes work and affect a distribution. See @Toy-model_1.

Next toy model. We have a simple process x_1 = x_0 + c + \eta, where c is a constant and \eta is gaussian noise. x_0 is considerd a sample from a multimodal complex distribution p0(x), namely the starting data distribtion. The process determiens the evolution of the system via the internal dynamic time t=0,1,2,... . 

In turn, we define the joint probability distribution p(x0,x1) as the product of the prior and the gaussian transition kernel. Finally, we can compute the score of this joint probability: take the logarithm, simplify, and then take the partial derivartive wrt to x0 and x1. This gives a two dimentional score. 

Your task: in the style of /Users/marcolomele/Documents/Repos/Score_Diffusion/Toy-models/Toy-model_1/ou_diffusion_explainer.html, create another beautiful, interactive, and concise report on this toy model. At the end of the html, create two animations:

    1. evolution of shape of distribution of system as it moves over the x axis due to the process being applied repeatedly. 

    2. plot of vector field field for different values of x0 and x1.

Assume that p0 is trimodal as mixture between 3 gaussians.

