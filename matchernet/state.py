import numpy as np

from matchernet import utils


class State(object):
    """Class State is a state handler that maintains, serializes, and deserializes the state of Bundles or Matchers.
    The methods serialize() and deserialize() are required for BriCA1 components to exchange their states as numpy.array objects.

    A Bundle/Matcher has its state as a dictionary.
    For exmaple,
    B0.state = state.State()
    B0.state.data = {"mu":np.array([1,2]),
               "Sigma":np.array([[1,0],[0,1]])}
    The disctionary is serialized with a method  serialize()
    q = B0.serialize()
    into a numpy.array object  q  .
    The serialized array is exchanged through  BriCA1 IN/OUT ports
    and deserialized with a method  deserialize() as.
    B0.deserialize(q)
    """

    def __init__(self, mu):
        self.data = {"mu": mu}


class StatePlain(State):
    """StatePlain is a State that handles plain numpy.array.
    """

    def __init__(self, mu):
        """Initializer takes a dimensionarity of the vector.
        """
        super(StatePlain, self).__init__(mu)


class StateMuSigmaDiag(State):
    """StateMuSigmaDiag is a state handler that handles
    state variable as the following dictionary style.
    B.state.data = {"id":1,
            "mu":numpy.array([1,2,3]),
            "sigma":numpy.array([2.0,2.0,2.0])}
    Note that StateMuSigma and StateMuSigmaDiag have
        n x n matrix "Sigma" and n vector "sigma",
        respectively.
    """

    def __init__(self, mu, sigma):
        super(StateMuSigmaDiag, self).__init__(mu)
        self.data["sigma"] = sigma


class StateMuSigma(State):
    """StateMuSigma is a state handler that handles
    state variable as the following dictionary style.
    B.state.data = {"id":1,
            "mu":numpy.array([1,2,3]),
            "Sigma":numpy.array([[1,0,0],[0,1,0],[0,0,1]])}
    Note that StateMuSigma and StateMuSigmaDiag have
        n x n matrix "Sigma" and n vector "sigma",
        respectively.
    """

    def __init__(self, mu, Sigma):
        super(StateMuSigma, self).__init__(mu)
        self.data["Sigma"] = Sigma
