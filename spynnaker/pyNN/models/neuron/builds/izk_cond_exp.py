from spynnaker.pyNN.models.neuron.input_types.input_type_conductance \
    import InputTypeConductance
from spynnaker.pyNN.models.neuron.neuron_models.neuron_model_izh \
    import NeuronModelIzh
from spynnaker.pyNN.models.neuron.synapse_types.synapse_type_exponential \
    import SynapseTypeExponential
from spynnaker.pyNN.models.neuron.threshold_types.threshold_type_static \
    import ThresholdTypeStatic
from spynnaker.pyNN.models.neuron.abstract_population_vertex \
    import AbstractPopulationVertex

_IZK_THRESHOLD = 30.0


class IzkCondExp(AbstractPopulationVertex):

    _model_based_max_atoms_per_core = 255

    default_parameters = {
        'a': 0.02, 'c': -65.0, 'b': 0.2, 'd': 2.0, 'i_offset': 0,
        'u_init': -14.0, 'v_init': -70.0, 'tau_syn_E': 5.0, 'tau_syn_I': 5.0,
        'e_rev_E': 0.0, 'e_rev_I': -70.0}

    # noinspection PyPep8Naming
    def __init__(
            self, n_neurons, machine_time_step, timescale_factor,
            spikes_per_second=None, ring_buffer_sigma=None, constraints=None,
            label=None,
            a=default_parameters['a'], b=default_parameters['b'],
            c=default_parameters['c'], d=default_parameters['d'],
            i_offset=default_parameters['i_offset'],
            u_init=default_parameters['u_init'],
            v_init=default_parameters['v_init'],
            tau_syn_E=default_parameters['tau_syn_E'],
            tau_syn_I=default_parameters['tau_syn_I'],
            e_rev_E=default_parameters['e_rev_E'],
            e_rev_I=default_parameters['e_rev_I']):

        neuron_model = NeuronModelIzh(
            n_neurons, machine_time_step, a, b, c, d, v_init, u_init, i_offset)
        synapse_type = SynapseTypeExponential(
            n_neurons, machine_time_step, tau_syn_E, tau_syn_I)
        input_type = InputTypeConductance(n_neurons, e_rev_E, e_rev_I)
        threshold_type = ThresholdTypeStatic(n_neurons, _IZK_THRESHOLD)

        AbstractPopulationVertex.__init__(
            self, n_neurons=n_neurons, binary="IZK_cond_exp.aplx", label=label,
            max_atoms_per_core=IzkCondExp._model_based_max_atoms_per_core,
            machine_time_step=machine_time_step,
            timescale_factor=timescale_factor,
            spikes_per_second=spikes_per_second,
            ring_buffer_sigma=ring_buffer_sigma,
            model_name="IZK_cond_exp", neuron_model=neuron_model,
            input_type=input_type, synapse_type=synapse_type,
            threshold_type=threshold_type, constraints=constraints)

    @staticmethod
    def set_model_max_atoms_per_core(new_value):
        IzkCondExp._model_based_max_atoms_per_core = new_value
