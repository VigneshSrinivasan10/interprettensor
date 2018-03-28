# Interpret Tensor - Slim TF wrapper to compute LRP

The Layer-wise Relevance Propagation (LRP) algorithm explains a classifer's prediction specific to a given data point by attributing relevance scores to important components of the input by using the topology of the learned model itself.

This tensorflow wrapper provides simple and accessible stand-alone implementations of LRP for artificial neural networks.

<img src="doc/images/1.png" width="215" height="215"> <img src="doc/images/2.png" width="215" height="215"> <img src="doc/images/3.png" width="215" height="215"> <img src="doc/images/4.png" width="215" height="215">

### Requirements
    tensorflow >= 1.0.0
    python >= 3
    matplotlib >= 1.3.1
    scikit-image > 0.11.3
    
# Features

## 1. Model 

This TF-wrapper considers the layers in the neural network to be in the form of a Sequence. A quick way to define a network would be

        net = Sequential([Linear(input_dim=784,output_dim=1296, act ='relu', batch_size=FLAGS.batch_size),
                     Linear(1296, act ='relu'), 
                     Linear(1296, act ='relu'),
                     Linear(10, act ='relu'),
                     Softmax()])

        output = net.forward(input_data)
             
## 2. Train the network

This `net` can then be used to propogate and optimize using

        trainer = net.fit(output, ground_truth, loss='softmax_crossentropy', optimizer='adam', opt_params=[FLAGS.learning_rate])

## 3. LRP - Layer-wise relevance propagation

And compute the contributions of the input pixels towards the decision by

        relevance = net.lrp(output, 'simple', 1.0)

the different lrp variants available are:

        'simple'and 'epsilon','flat','ww' and 'alphabeta' 

## 4. Compute relevances every layer backwards from the output to the input  

Follow steps (1) from Features mentioned above.

       relevance_layerwise = []
       R = output
       for layer in net.modules[::-1]:
           R = net.lrp_layerwise(layer, R, 'simple')
           relevance_layerwise.append(R)
           
# Examples 

To run the given mnist examples,
   
        cd examples
        python mnist_linear.py --relevance=True

It downloads and extract the mnist datset, runs it on a neural netowrk and plots the relevances once the network is optimized. The relvances of the images can be viewed on the tensorboard using
   
        tensorboard --logdir=mnist_linear_logs

# LRP for a pretrained model

Follow steps (1) and (3) from Features mentioned above.


# The LRP Toolbox Paper

When using (any part) of this wrapper, please cite [our paper](http://jmlr.org/papers/volume17/15-618/15-618.pdf)

    @article{JMLR:v17:15-618,
        author  = {Sebastian Lapuschkin and Alexander Binder and Gr{{\'e}}goire Montavon and Klaus-Robert M{{{\"u}}}ller and Wojciech Samek},
        title   = {The LRP Toolbox for Artificial Neural Networks},
        journal = {Journal of Machine Learning Research},
        year    = {2016},
        volume  = {17},
        number  = {114},
        pages   = {1-5},
        url     = {http://jmlr.org/papers/v17/15-618.html}
    }


    
# Misc

For further research and projects involving LRP, visit [heatmapping.org](http://heatmapping.org)
   
