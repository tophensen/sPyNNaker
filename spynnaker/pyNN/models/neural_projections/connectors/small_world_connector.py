from spynnaker.pyNN.models.neural_projections.connectors.abstract_connector \
    import AbstractConnector
from spynnaker.pyNN import exceptions


class SmallWorldConnector(AbstractConnector):

    def generate_synapse_list(self, presynaptic_population, 
                                    postsynaptic_population, 
                                    delay_scale, synapse_type):
        raise exceptions.SpynnakerException("This connector is currently not "
                                            "supported by the tool chain....."
                                            "watch this space")

    def __init__(self):
        raise exceptions.SpynnakerException("This connector is currently not "
                                            "supported by the tool chain....."
                                            "watch this space")
