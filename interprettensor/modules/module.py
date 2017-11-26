'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf


# -------------------------------
# Modules for the neural network
# -------------------------------
layer_count = 0
class Module:
    '''
    Superclass for all computation layer implementations
    '''

    def __init__(self):
        ''' The constructor '''
        global layer_count
        layer_count = layer_count + 1
        
        if hasattr(self, 'name'):
            self.name = self.name+'_'+str(layer_count)

            #self.previous_input=self.
            #print 'LAYER COUNT: '+str(layer_count)
        #values for presetting lrp decomposition parameters per layer
        self.lrp_var = None
        self.lrp_param = 1.
        #self.input_shape = self.input_shape
        # 
        # self.forward(self.input_tensor)
        
    def forward(self,X):
        ''' forward passes the input data X to the layer's output neurons '''
        return X

    def clean(self):
        ''' clean can be used to remove any temporary variables from the layer, e.g. just before serializing the layer object'''
        pass



    def set_lrp_parameters(self,lrp_var=None,param=None):
        ''' pre-sets lrp parameters to use for this layer. see the documentation of Module.lrp for details '''
        self.lrp_var = lrp_var
        self.lrp_param = param

    def lrp(self,R, lrp_var=None,param=None):
        '''
        Performs LRP by calling subroutines, depending on lrp_var and param or
        preset values specified via Module.set_lrp_parameters(lrp_var,lrp_param)

        If lrp parameters have been pre-specified (per layer), the corresponding decomposition
        will be applied during a call of lrp().

        Specifying lrp parameters explicitly when calling lrp(), e.g. net.lrp(R,lrp_var='alpha',param=2.),
        will override the preset values for the current call.

        How to use:

        net.forward(X) #forward feed some data you wish to explain to populat the net.

        then either:

        net.lrp() #to perform the naive approach to lrp implemented in _simple_lrp for each layer

        or:

        for m in net.modules:
            m.set_lrp_parameters(...)
        net.lrp() #to preset a lrp configuration to each layer in the net

        or:

        net.lrp(somevariantname,someparameter) # to explicitly call the specified parametrization for all layers (where applicable) and override any preset configurations.

        Parameters
        ----------

        R : numpy.ndarray
            relevance input for LRP.
            should be of the same shape as the previously produced output by <Module>.forward

        lrp_var : str
            either 'none' or 'simple' or None for standard Lrp ,
            'epsilon' for an added epsilon slack in the denominator
            'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
            'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
            'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.

        param : double
            the respective parameter for the lrp method of choice

        Returns
        -------
        R : the backward-propagated relevance scores.
            shaped identically to the previously processed inputs in <Module>.forward
        '''

        if lrp_var == None and param == None:
            # module.lrp(R) has been called without further parameters.
            # set default values / preset values
            lrp_var = self.lrp_var
            param = self.lrp_param

        if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
            with tf.name_scope(self.name+'_simple_relevance'):
                Rx = self._simple_lrp(R)
            tf.summary.histogram(self.name, Rx)
            return Rx
        elif lrp_var.lower() == 'flat':
            return self._flat_lrp(R)
        elif lrp_var.lower() == 'ww' or lrp_var.lower() == 'w^2':
            return self._ww_lrp(R)
        elif lrp_var.lower() == 'epsilon':
            with tf.name_scope(self.name+'_epsilon_relevance'):
                Rx = self._epsilon_lrp(R,param)
            tf.summary.histogram(self.name, Rx)
            return Rx
        elif lrp_var.lower() == 'alphabeta' or lrp_var.lower() == 'alpha':
            return self._alphabeta_lrp(R,param)
        else:
            print('Unknown lrp variant', lrp_var)


    # ---------------------------------------------------------
    # Methods below should be implemented by inheriting classes
    # ---------------------------------------------------------

    def _simple_lrp(self,R):
        raise NotImplementedError()

    def _flat_lrp(self,R):
        raise NotImplementedError()

    def _ww_lrp(self,R):
        raise NotImplementedError()

    def _epsilon_lrp(self,R,param):
        raise NotImplementedError()

    def _alphabeta_lrp(self,R,param):
        raise NotImplementedError()
