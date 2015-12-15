from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.neural_properties.randomDistributions\
    import generate_parameter
from pacman.model.partitionable_graph.abstract_partitionable_vertex \
    import AbstractPartitionableVertex
from pacman.model.constraints.key_allocator_constraints\
    .key_allocator_contiguous_range_constraint \
    import KeyAllocatorContiguousRangeContraint

from spynnaker.pyNN.models.common.abstract_spike_recordable \
    import AbstractSpikeRecordable
from spynnaker.pyNN.models.common.spike_recorder import SpikeRecorder
from spynnaker.pyNN.utilities.conf import config

from spinn_front_end_common.abstract_models.abstract_data_specable_vertex\
    import AbstractDataSpecableVertex
#from spinn_front_end_common.abstract_models.\
#    abstract_outgoing_edge_same_contiguous_keys_restrictor import \
#    OutgoingEdgeSameContiguousKeysRestrictor
# this might or might not be needed to subclass
from spinn_front_end_common.abstract_models.\
    abstract_provides_outgoing_edge_constraints import \
    AbstractProvidesOutgoingEdgeConstraints
from spinn_front_end_common.interface.buffer_management\
    .buffer_models.abstract_receive_buffers_to_host \
    import AbstractReceiveBuffersToHost

from data_specification.data_specification_generator\
    import DataSpecificationGenerator
from data_specification.enums.data_type import DataType

from enum import Enum
import math
import numpy
import logging

logger = logging.getLogger(__name__)

PARAMS_BASE_WORDS = 4
PARAMS_REMOTE_WORDS = 6
PARAMS_WORDS_PER_NEURON = 3
RANDOM_SEED_WORDS = 4


class SpikeSourceRemote(
        AbstractSpikeRecordable, AbstractPartitionableVertex,
        AbstractDataSpecableVertex,AbstractProvidesOutgoingEdgeConstraints,
        AbstractReceiveBuffersToHost):
    """
    This class represents a Poisson Spike source object, which can represent
    a pynn_population.py of virtual neurons each with its own parameters.
    """

#    CORE_APP_IDENTIFIER = constants.SPIKESOURCEREMOTE_CORE_APPLICATION_ID
    _REMOTE_SPIKE_SOURCE_REGIONS = Enum(
        value="_REMOTE_SPIKE_SOURCE_REGIONS",
        names=[('SYSTEM_REGION', 0),
               ('POISSON_PARAMS_REGION', 1),
               ('REMOTE_PARAMS_REGION', 2),
               ('SPIKE_HISTORY_REGION', 3),
               ('BUFFERING_OUT_STATE', 4)])
               
    _N_POPULATION_RECORDING_REGIONS = 1
    
    _model_based_max_atoms_per_core = 256

    def __init__(self, n_neurons, machine_time_step, timescale_factor,
                 constraints=None, label="SpikeSourceRemote",
                 rate=1.0, seed=None,
                 min_rate=0.01, max_rate=100.0, sensor_min=-20,
                 sensor_max=2048, gauss_width=1.0):
        """
        Creates a new SpikeSourceRemote Object.
        """
        AbstractPartitionableVertex.__init__(
            self, n_atoms=n_neurons, label=label, constraints=constraints,
            max_atoms_per_core=self._model_based_max_atoms_per_core)
        AbstractDataSpecableVertex.__init__(
            self, machine_time_step=machine_time_step,
            timescale_factor=timescale_factor)
        AbstractSpikeRecordable.__init__(self)
        AbstractReceiveBuffersToHost.__init__(self)

        self._rate = rate
        self._seed = seed
        self.max_rate = max_rate
        self.min_rate = min_rate
        # transform gauss_width from neuron-coordinates to sensor units
        gauss_width = (sensor_max-sensor_min)/(n_neurons-1.)*gauss_width
        self.gauss_width = int(round(gauss_width))
        sensor_min,sensor_max = int(round(sensor_min)),int(round(sensor_max))
        self.sensor_min = sensor_min # these might get scaled
        self.sensor_max = sensor_max # at (or right before) runtime
        
        self._rng = numpy.random.RandomState(seed)

        # Prepare for recording, and to get spikes
        self._spike_recorder = SpikeRecorder(machine_time_step)
        self._spike_buffer_max_size = config.getint(
            "Buffers", "spike_buffer_size")
        self._buffer_size_before_receive = config.getint(
            "Buffers", "buffer_size_before_receive")
        self._time_between_requests = config.getint(
            "Buffers", "time_between_requests")

        
    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate):
        self._rate = rate

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
      
    @property
    def model_name(self):
        """
        Return a string representing a label for this class.
        """
        return "SpikeSourceRemote"

    @staticmethod
    def set_model_max_atoms_per_core(new_value):
        """

        :param new_value:
        :return:
        """
        self.\
            _model_based_max_atoms_per_core = new_value


    @staticmethod
    def get_params_bytes(vertex_slice):
        """
        Gets the size of the parameters in bytes
        :param vertex_slice:
        """
        return (RANDOM_SEED_WORDS + PARAMS_BASE_WORDS + PARAMS_REMOTE_WORDS +
                (((vertex_slice.hi_atom - vertex_slice.lo_atom) + 1) *
                 PARAMS_WORDS_PER_NEURON)) * 4

    def reserve_memory_regions(self, spec, setup_sz, poisson_params_sz,
                               spike_hist_buff_sz):
        """
        Reserve memory regions for poisson source parameters
        and output buffer.
        :param spec:
        :param setup_sz:
        :param poisson_params_sz:
        :param spike_hist_buff_sz:
        :return:
        """
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=self._REMOTE_SPIKE_SOURCE_REGIONS.SYSTEM_REGION.value,
            size=setup_sz, label='setup')
        spec.reserve_memory_region(
            region=self._REMOTE_SPIKE_SOURCE_REGIONS
                       .POISSON_PARAMS_REGION.value,
            size= ( poisson_params_sz - PARAMS_REMOTE_WORDS*4 ), label='PoissonParams')
        spec.reserve_memory_region(
            region=self._REMOTE_SPIKE_SOURCE_REGIONS
                       .REMOTE_PARAMS_REGION.value,
            size=PARAMS_REMOTE_WORDS*4, label='RemoteParams')
        self.reserve_buffer_regions(
            spec, self._REMOTE_SPIKE_SOURCE_REGIONS.BUFFERING_OUT_STATE.value,
            [self._REMOTE_SPIKE_SOURCE_REGIONS.SPIKE_HISTORY_REGION.value],
            [spike_hist_buff_sz])

    def _write_setup_info(self, spec, spike_history_region_sz, ip_tags,
            buffer_size_before_receive):
        """ Write information used to control the simulation and gathering of\
            results.
        :param spec:
        :param spike_history_region_sz:
        :param ip_rags
        :return:
        """

        self._write_basic_setup_info(
            spec, self._REMOTE_SPIKE_SOURCE_REGIONS.SYSTEM_REGION.value)
        self.write_recording_data(
            spec, ip_tags, [spike_history_region_sz],
            buffer_size_before_receive, self._time_between_requests)


    def _write_remote_parameters(self, spec, num_neurons):
        """
        Generate Neuron Parameter data for remote control (region 3)
        """
        spec.comment("\nWriting Neuron Remote Parameters for {} remote sources:\n"
                     .format(num_neurons))

        # Set the focus to the memory region 2 (neuron parameters):
        spec.switch_write_focus(
            region=self._REMOTE_SPIKE_SOURCE_REGIONS
                       .REMOTE_PARAMS_REGION.value)
        spec.write_value(data=self.sensor_min, data_type=DataType.INT32)
        spec.write_value(data=self.sensor_max, data_type=DataType.INT32)

        #TODO: this would make more sense with integers and inverse rates        
        p_min = 1000000.0 / (self.max_rate * self._machine_time_step)
        if p_min > 32767:
            p_min = 32767
        p_max = 1000000.0 / (self.min_rate * self._machine_time_step)
        if p_max > 32767:
            p_max = 32767
        spec.write_value(p_min, data_type=DataType.S1615)
        spec.write_value(p_max, data_type=DataType.S1615)
        
        spec.write_value(self.gauss_width, data_type=DataType.UINT32)
        spec.write_value(self.n_atoms, data_type=DataType.UINT32)

    def _write_poisson_parameters(self, spec, key, num_neurons):
        """
        Generate Neuron Parameter data for Remote spike sources (region 2):
        :param spec:
        :param key:
        :param num_neurons:
        :return:
        """
        spec.comment("\nWriting Neuron Parameters for {} remote sources:\n"
                     .format(num_neurons))

        # Set the focus to the memory region 2 (neuron parameters):
        spec.switch_write_focus(
            region=self._REMOTE_SPIKE_SOURCE_REGIONS
                       .POISSON_PARAMS_REGION.value)

        # Write header info to the memory region:

        # Write Key info for this core:
        if key is None:
            # if theres no key, then two falses will cover it.
            spec.write_value(data=0)
            spec.write_value(data=0)
        else:
            # has a key, thus set has key to 1 and then add key
            spec.write_value(data=1)
            spec.write_value(data=key)

        # Write the random seed (4 words), generated randomly!
        spec.write_value(data=self._rng.randint(0x7FFFFFFF))
        spec.write_value(data=self._rng.randint(0x7FFFFFFF))
        spec.write_value(data=self._rng.randint(0x7FFFFFFF))
        spec.write_value(data=self._rng.randint(0x7FFFFFFF))

        slow_sources = list()
        for i in range(0, num_neurons):

            # Get the parameter values for source i:
            rate_val = generate_parameter(self._rate, i)
            slow_sources.append([i, rate_val])

        # Write the numbers of each type of source
        spec.write_value(data=len(slow_sources))
        spec.write_value(data=0)

        # Now write one struct for each slow source as follows
        #
        #   typedef struct slow_spike_source_t
        #   {
        #     uint32_t neuron_id;
        #
        #     accum mean_isi_ticks;
        #     accum time_to_spike_ticks;
        #   } slow_spike_source_t;
        for (neuron_id, rate_val) in slow_sources:
            isi_val = float(1000000.0 / (rate_val * self._machine_time_step))
            spec.write_value(data=neuron_id, data_type=DataType.UINT32)
            spec.write_value(data=isi_val, data_type=DataType.S1615)
            spec.write_value(data=0x0, data_type=DataType.UINT32)

    def is_recording_spikes(self):
        return self._spike_recorder.record

    def set_recording_spikes(self):
        ip_address = config.get("Buffers", "receive_buffer_host")
        port = config.getint("Buffers", "receive_buffer_port")
        self.set_buffering_output(ip_address, port)
        self._spike_recorder.record = True

    def get_spikes(self, placements, graph_mapper):
        return self._spike_recorder.get_spikes(
            self._label, self.buffer_manager,
            self._REMOTE_SPIKE_SOURCE_REGIONS.SPIKE_HISTORY_REGION.value,
            self._REMOTE_SPIKE_SOURCE_REGIONS.BUFFERING_OUT_STATE.value,
            placements, graph_mapper, self)

    def get_outgoing_edge_constraints(self, partitioned_edge, graph_mapper):
        """
        gets the constraints for edges going out of this vertex
        :param partitioned_edge: the parittioned edge that leaves this vertex
        :param graph_mapper: the graph mapper object
        :return: list of constraints
        """
        return [KeyAllocatorContiguousRangeContraint()]

    def is_data_specable(self):
        """
        helper method for isinstance
        :return:
        """
        return True

    def is_receives_buffers_to_host(self):
        return True

    def get_value(self, key):
        """ Get a property of the overall model
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise Exception("Population {} does not have parameter {}".format(
            self, key))
                        
    # inherited from partitionable vertex
    def get_sdram_usage_for_atoms(self, vertex_slice, graph):
        """
        method for calculating SDRAM usage
        :param vertex_slice:
        :param graph:
        :return:
        """
        poisson_params_sz = self.get_params_bytes(vertex_slice)
        spike_hist_buff_sz = min((
            self._spike_recorder.get_sdram_usage_in_bytes(
                vertex_slice.n_atoms, self._no_machine_time_steps),
            self._spike_buffer_max_size))
        return ((constants.DATA_SPECABLE_BASIC_SETUP_INFO_N_WORDS * 4) +
                self.get_recording_data_size(1) +
                self.get_buffer_state_region_size(1) +
                poisson_params_sz + spike_hist_buff_sz)

    def get_dtcm_usage_for_atoms(self, vertex_slice, graph):
        """
        method for calculating dtcm usage for a collection of atoms
        :param vertex_slice:
        :param graph:
        :return:
        """
        return 512 * 4 # the current lutsize

    def get_cpu_usage_for_atoms(self, vertex_slice, graph):
        """
        Gets the CPU requirements for a range of atoms

        :param vertex_slice:
        :param graph:
        :return:
        """
        return 0

    # inherited from dataspecable vertex
    def generate_data_spec(self, subvertex, placement, subgraph, graph,
                           routing_info, hostname, graph_mapper, report_folder,
                           ip_tags, reverse_ip_tags, write_text_specs,
                           application_run_time_folder):
        """
        Model-specific construction of the data blocks necessary to build a
        single SpikeSourcePoisson on one core.
        :param subvertex:
        :param placement:
        :param subgraph:
        :param graph:
        :param routing_info:
        :param hostname:
        :param graph_mapper:
        :param report_folder:
        :param ip_tags:
        :param reverse_ip_tags:
        :param write_text_specs:
        :param application_run_time_folder:
        :return:
        """
        data_writer, report_writer = \
            self.get_data_spec_file_writers(
                placement.x, placement.y, placement.p, hostname, report_folder,
                write_text_specs, application_run_time_folder)

        spec = DataSpecificationGenerator(data_writer, report_writer)

        vertex_slice = graph_mapper.get_subvertex_slice(subvertex)

        spike_hist_buff_sz = self._spike_recorder.get_sdram_usage_in_bytes(
            vertex_slice.n_atoms, self._no_machine_time_steps)
        buffer_size_before_receive = self._buffer_size_before_receive
        if config.getboolean("Buffers", "enable_buffered_recording"):
            if spike_hist_buff_sz < self._spike_buffer_max_size:
                buffer_size_before_receive = spike_hist_buff_sz + 256
            else:
                spike_hist_buff_sz = self._spike_buffer_max_size
        else:
            buffer_size_before_receive = spike_hist_buff_sz + 256

        spec.comment("\n*** Spec for SpikeSourceRemote Instance ***\n\n")

        # Basic setup plus 8 bytes for recording flags and recording size
        setup_sz = ((constants.DATA_SPECABLE_BASIC_SETUP_INFO_N_WORDS * 4) +
                    self.get_recording_data_size(1))
                    
        poisson_params_sz = self.get_params_bytes(vertex_slice)

        # Reserve SDRAM space for memory areas:
        self.reserve_memory_regions(
            spec, setup_sz, poisson_params_sz, spike_hist_buff_sz)

        self._write_setup_info(
            spec, spike_hist_buff_sz, ip_tags, buffer_size_before_receive)

        # Every subedge should have the same key
        key = None
        subedges = subgraph.outgoing_subedges_from_subvertex(subvertex)
        if len(subedges) > 0:
            keys_and_masks = routing_info.get_keys_and_masks_from_subedge(
                subedges[0])
            key = keys_and_masks[0].key

        self._write_poisson_parameters(spec, key, vertex_slice.n_atoms)
        self._write_remote_parameters(spec, vertex_slice.n_atoms)

        # End-of-Spec:
        spec.end_specification()
        data_writer.close()

        return [data_writer.filename]

    def get_binary_file_name(self):
        """

        :return:
        """
        return "spike_source_remote.aplx"

    def is_recordable(self):
        """

        :return:
        """
        return True

    def is_data_specable(self):
        """
        helper method for isinstance
        :return:
        """
        return True
        
    def recieves_multicast_commands(self):
        return True
        
        
class SpikeSourceDeterministicRate(SpikeSourceRemote):
        
    def get_binary_file_name(self):
        """

        :return:
        """
        return "spike_source_remote.aplx"

class SpikeSourcePoissonRate(SpikeSourceRemote):

    def get_binary_file_name(self):
        """

        :return:
        """
        return "spike_source_remote_poisson.aplx"
        
        
class SpikeSourcePoissonRBF(SpikeSourceRemote):

    def get_binary_file_name(self):
        """

        :return:
        """
        return "spike_source_remote_poisson_rbf.aplx"

class SpikeSourceDeterministicRBF(SpikeSourceRemote):

    def get_binary_file_name(self):
        """

        :return:
        """
        return "spike_source_remote_rbf.aplx"
        
