/*! \file
 *
 *  \brief This file contains the main functions for a remote controlled spike generator.
 *
 *
 */

#include "../../common/out_spikes.h"
#include "../../common/maths-util.h"

#include <math.h>

#include <data_specification.h>
#include <recording.h>
#include <debug.h>
#include <simulation.h>
// this could be ifdefd/factored out, if it were not to seed_size!
#include <random.h>
#include <spin1_api.h>

//! data structure for spikes which have multiple timer tick between firings
//! this is separated from spikes which fire at least once every timer tick as
//! there are separate algorithms for each type.
typedef struct slow_spike_source_t {
    uint32_t neuron_id;
    REAL mean_isi_ticks;
    REAL time_to_spike_ticks;
} slow_spike_source_t;
/*
//! spike source array region ids in human readable form
typedef enum region{
    system, poisson_params, remote_params, spike_history,
}region;
*/
typedef enum region{
    SYSTEM, POISSON_PARAMS, REMOTE_PARAMS,
    BUFFERING_OUT_SPIKE_RECORDING_REGION,
    BUFFERING_OUT_CONTROL_REGION
}region;

#define NUMBER_OF_REGIONS_TO_RECORD 1


//! what each position in the poisson parameter region actually represent in
//! terms of data (each is a word)
typedef enum poisson_region_parameters{
    HAS_KEY, TRANSMISSION_KEY, PARAMETER_SEED_START_POSITION,
} poisson_region_parameters;

// Globals
//! global variable which contains all the data for neurons which are expected
//! to exhibit slow spike generation (less than 1 per timer tick)
//! (separated for efficiently purposes)
static slow_spike_source_t *slow_spike_source_array = NULL;
//! counter for how many neurons exhibit slow spike generation
static uint32_t num_slow_spike_sources = 0;
//! a variable that will contain the seed to initiate the poisson generator.
static mars_kiss64_seed_t spike_source_seed;
//! a vairable which checks if there has been a key allocated to this spike
//! source posson
static bool has_been_given_key;
//! A variable that contains the key value that this model should transmit with
static uint32_t key;
//! keeps track of which types of recording should be done to this model.
static uint32_t recording_flags = 0;
//! the time interval parameter TODO this variable could be removed and use the
//! timer tick callback timer value.
static uint32_t time;
//! the number of timer tics that this model should run for before exiting.
static uint32_t simulation_ticks = 0;
//! the int that represnets the bool for if the run is infinte or not.
static uint32_t infinite_run;

static REAL gauss_width = 20.;
static REAL period_min = 0.; // 1 / (maximal rate) in units of ticks
static REAL period_max = 0.;       // 1 / (minimal rate) in units of ticks
static REAL *lu_table = NULL; 
#define LUTSIZE 512
static REAL sensorvalue_to_lutidx;
static REAL sensorvalue_to_neuidx;
static REAL neuidx_to_sensorvalue;

static int32_t sensor_max = + 1444;
static int32_t sensor_min = - 256;
static int32_t sensor_range = 1700;
static uint32_t n_max = 1;
static uint32_t input_shift = 0; 

//! \deduces the time in timer ticks until the next spike is to occur given a
//! mean_isi_ticks
//! \param[in] mean_inter_spike_interval_in_ticks The mean number of ticks
//! before a spike is expected to occur in a slow process.
//! \return a real which represents time in timer ticks until the next spike is
//! to occur
static inline REAL slow_spike_source_get_time_to_spike(
        const REAL mean_inter_spike_interval_in_ticks) {
    return exponential_dist_variate(mars_kiss64_seed, spike_source_seed)
            * mean_inter_spike_interval_in_ticks;
}

#ifdef RBF
// fills the previously allocated lu_table, a look-up table
// for gaussian receiptive fields of the remote controlled neurons
// the final index corresponds to an argument of true_lutcut
void fill_lut(void) {
  // yes, we are using floats to build our LUT
  float step = 1.17741 * 3.0 / LUTSIZE; // = HWHM / ( LUTSIZE / 3 )
  float arg = 0.0;                             // so lut finishes at 0.002
  for ( uint i = 0 ; i < LUTSIZE; i++) {
    arg = step * i;
    arg = expf( (arg*arg)*0.5 ) * period_min;
    if (arg < period_max) {
        lu_table[i] = arg;
        }
    else lu_table[i] = period_max;;
  }

  // LUTSIZE / 3.0 is the (broken) index at which to find 0.5
  sensorvalue_to_lutidx = LUTSIZE ;
  sensorvalue_to_lutidx /= (gauss_width * 3.0);
}

// Looks up a given value in our gauss LUT
static inline REAL lookup(const REAL value) {
// this function is symmetric -> lookup(value) = lookup(-value)
     uint idx;     
     if (value < 0) idx = - value * sensorvalue_to_lutidx + 0.5k;
     else idx = value * sensorvalue_to_lutidx + 0.5k; 
     if ( idx >= LUTSIZE ) return period_max; // outside of range
     return lu_table[idx];
}
#endif

#ifndef RBF
void fill_lut(void) {
    float step = 1.0f * sensor_range / LUTSIZE;
    float arg = 0.0f;
    for (uint i = 0 ; i < LUTSIZE ; i++) {
        arg = step*i;
        arg = 1.0f * period_min * period_max * sensor_range / 
              ((period_max - period_min)*(arg - sensor_min) + period_min*(sensor_max - sensor_min));
        lu_table[i] = arg;
        }
    sensorvalue_to_lutidx = (1.0f / step);
    }

static REAL lookup(const int value) {
    uint idx;
    idx = value * sensorvalue_to_lutidx + 0.5k;
    if (idx >= LUTSIZE) return period_min;
    else return lu_table[idx];
}
#endif


void incoming_update_callback(uint key, uint payload) {
    use(key);
    int * signed_payload = (int*) &payload;
    REAL value, new_tts;
    payload = payload >> input_shift;
    { 
        *signed_payload -= sensor_min;
        *signed_payload = min(sensor_range,*signed_payload);
#ifndef RBF        
        value = lookup(*signed_payload);
#endif
        for (uint8_t i=0; i<num_slow_spike_sources; i++) {
            slow_spike_source_t *slow_spike_source = &slow_spike_source_array[i];
#ifdef RBF
            // lookup at distance of this neuron to setpoint (in sensor units)
            value = lookup(( slow_spike_source->neuron_id * neuidx_to_sensorvalue ) 
                            - *signed_payload ); 
#endif
            new_tts = (slow_spike_source->time_to_spike_ticks / slow_spike_source->mean_isi_ticks) * value;

            if (new_tts > REAL_CONST(0.0)) { // compiler error with REAL_COMPARE
                slow_spike_source->time_to_spike_ticks = new_tts;
            }
            
            slow_spike_source->mean_isi_ticks = value; // Update time to spike buffer
        }
    }
}


//! \The callback used when a timer tick interrupt is set off. The result of
//! this is to transmit any spikes that need to be sent at this timer tick,
//! update any recording, and update the state machine's states.
//! If the timer tick is set to the end time, this method will call the
//! spin1api stop command to allow clean exit of the executable.
//! \param[in] timer_count the number of times this call back has been
//! executed since start of simulation
//! \param[in] unused for consistency sake of the API always returning two
//! parameters, this parameter has no semantics currently and thus is set to 0
//! \return None
void timer_callback(uint timer_count, uint unused) {
    use(timer_count);
    use(unused);
    time++;

    log_debug("Timer tick %u", time);

    // If a fixed number of simulation ticks are specified and these have passed
    if (infinite_run != TRUE && time >= simulation_ticks) {
        log_debug("Simulation complete.\n");

        // Finalise any recordings that are in progress, writing back the final
        // amounts of samples recorded to SDRAM
        if (recording_flags > 0) {
            recording_finalise();
        }
        
        spin1_exit(0);
        return;
    }

    // Loop through slow spike sources
    slow_spike_source_t *slow_spike_sources = slow_spike_source_array;
    for (index_t s = num_slow_spike_sources; s > 0; s--) {

        // If this spike source is active this tick
        slow_spike_source_t *slow_spike_source = slow_spike_sources++;
        if ((time > 0) && (REAL_COMPARE(slow_spike_source->mean_isi_ticks, !=,
                    REAL_CONST(0.0)))) {

            // If this spike source should spike now
            if (REAL_COMPARE(slow_spike_source->time_to_spike_ticks, <=,
                             REAL_CONST(0.0))) {

                // Write spike to out spikes
                out_spikes_set_spike(slow_spike_source->neuron_id);

                // if no key has been given, do not send spike to fabric.
                if (has_been_given_key) {

                    // Send package
                    while (!spin1_send_mc_packet(
                            key | slow_spike_source->neuron_id, 0,
                            NO_PAYLOAD)) {
                        spin1_delay_us(1);
                    }
                    log_debug("Sending spike packet %x at %d\n",
                        key | slow_spike_source->neuron_id, time);
                }

                // Update time to spike
#ifdef POISSON
                slow_spike_source->time_to_spike_ticks +=
                    slow_spike_source_get_time_to_spike(
                        slow_spike_source->mean_isi_ticks);
#else
                slow_spike_source->time_to_spike_ticks +=
                    slow_spike_source->mean_isi_ticks;
#endif
            }

            // Subtract tick
            slow_spike_source->time_to_spike_ticks -= REAL_CONST(1.0);
        }
    }
    
    // Record output spikes if required
    if (recording_flags > 0) {
        out_spikes_record(0, time);
    }
    out_spikes_reset();
    
    if (recording_flags > 0) {
        recording_do_timestep_update(time);
    }
}

bool read_remote_parameters(address_t address) {
    log_info("read_remote_parameters: starting");
#ifdef RBF
    log_debug("RBF active");
#endif
#ifdef POISSON
    log_debug("Poisson active");
#endif
    
    sensor_min = (int32_t) address[0];
    sensor_max = (int32_t) address[1];
  
    spin1_memcpy(&period_min, &address[2], 4);
    spin1_memcpy(&period_max, &address[3], 4);
    gauss_width = (int32_t) address[4];
    n_max = (uint32_t) address[5];

    // this is a weird scaling hack, to easily cope with REAL limitations
    while ( sensor_min <= -32767 || sensor_max >= 32767 ) {
      sensor_min = sensor_min >> 1;
      sensor_max = sensor_max >> 1;
      gauss_width = gauss_width >> 1;
      input_shift++;
    }

    sensor_range = sensor_max - sensor_min;

    sensorvalue_to_neuidx = (REAL) (n_max - 1.0k) / sensor_range;
    neuidx_to_sensorvalue = (REAL) sensor_range / (n_max - 1);

    lu_table = (REAL *) spin1_malloc( LUTSIZE * sizeof(REAL) );
    fill_lut(); 

    log_info("sensor_to_neu: %k, neu_to_sens: %k, sensor_to_lut: %k, sensor_to_isi: %k", \
            sensorvalue_to_neuidx,neuidx_to_sensorvalue,sensorvalue_to_lutidx);
    log_debug("sensor_min: %d, sensor_max: %d, sensor_range: %d", \
            sensor_min, sensor_max, sensor_range);
    log_debug("min period: %k, max period: %k, width: %k", period_min, period_max, gauss_width);

    return true;
    }
    
//! \entry method for reading the parameters stored in poisson parameter region
//! \param[in] address the absolute SDRAm memory address to which the
//! poisson parameter region starts.
//! \return a boolean which is True if the parameters were read successfully or
//! False otherwise
bool read_poisson_parameters(address_t address) {

    log_debug("read_parameters: starting");

    has_been_given_key = address[HAS_KEY];
    key = address[TRANSMISSION_KEY];
    log_debug("\tkey = %08x", key);

    uint32_t seed_size = sizeof(mars_kiss64_seed_t) / sizeof(uint32_t);
    spin1_memcpy(spike_source_seed, &address[PARAMETER_SEED_START_POSITION],
        seed_size * sizeof(uint32_t));
    validate_mars_kiss64_seed(spike_source_seed);
    log_debug("\tSeed (%u) = %u %u %u %u", seed_size, spike_source_seed[0],
             spike_source_seed[1], spike_source_seed[2], spike_source_seed[3]);

    num_slow_spike_sources = address[PARAMETER_SEED_START_POSITION + seed_size];
    log_debug("\tslow spike sources = %u",
             num_slow_spike_sources);

    // Allocate DTCM for array of slow spike sources and copy block of data
    if (num_slow_spike_sources > 0) {
        slow_spike_source_array = (slow_spike_source_t*) spin1_malloc(
            num_slow_spike_sources * sizeof(slow_spike_source_t));
        if (slow_spike_source_array == NULL) {
            log_debug("Failed to allocate slow_spike_source_array");
            return false;
        }
        uint32_t slow_spikes_offset = PARAMETER_SEED_START_POSITION +
                                      seed_size + 2;
        spin1_memcpy(slow_spike_source_array,
                &address[slow_spikes_offset],
               num_slow_spike_sources * sizeof(slow_spike_source_t));

        // Loop through slow spike sources and initialise 1st time to spike
        for (int s = num_slow_spike_sources - 1 ; s >= 0; s--) {
            slow_spike_source_array[s].time_to_spike_ticks =
                slow_spike_source_get_time_to_spike(
                    slow_spike_source_array[s].mean_isi_ticks);
        }
    }

    log_info("read_parameters: completed successfully");
    return true;
}

//! \Initialises the model by reading in the regions and checking recording
//! data.
//! \param[in] *timer_period a pointer for the memory address where the timer
//! period should be stored during the function.
//! \return boolean of True if it successfully read all the regions and set up
//! all its internal data structures. Otherwise returns False
static bool initialize(uint32_t *timer_period) {
    log_debug("Initialise: started");

    // Get the address this core's DTCM data starts at from SRAM
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address)) {
        return false;
    }

    // Get the timing details
    address_t system_region = data_specification_get_region(
            SYSTEM, address);
    if (!simulation_read_timing_details(
            system_region, APPLICATION_NAME_HASH, timer_period,
            &simulation_ticks, &infinite_run)) {
        return false;
    }

    // Get the recording information
    uint8_t regions_to_record[] = {
        BUFFERING_OUT_SPIKE_RECORDING_REGION,
    };
    uint8_t n_regions_to_record = NUMBER_OF_REGIONS_TO_RECORD;
    uint32_t *recording_flags_from_system_conf =
        &system_region[SIMULATION_N_TIMING_DETAIL_WORDS];
    uint8_t state_region = BUFFERING_OUT_CONTROL_REGION;

    recording_initialize(
        n_regions_to_record, regions_to_record,
        recording_flags_from_system_conf, state_region, 2, &recording_flags);


    // Setup regions that specify spike source array data
    if (!read_poisson_parameters(
            data_specification_get_region(POISSON_PARAMS, address))) {
        return false;
    }
    if (!read_remote_parameters(
            data_specification_get_region(REMOTE_PARAMS, address))) {
        return false;
    }    

    log_debug("Initialise: completed successfully");

    return true;
}


//! \The only entry point for this model. it initialises the model, sets up the
//! Interrupts for the Timer tick and calls the spin1api for running.
void c_main(void) {
    // Load DTCM data
    uint32_t timer_period;
    if (!initialize(&timer_period)) {
        rt_error(RTE_SWERR);
    }

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // Initialise out spikes buffer to support number of neurons
    if (!out_spikes_initialize(num_slow_spike_sources)) {
         rt_error(RTE_SWERR);
    }

    // Set timer tick (in microseconds)
    spin1_set_timer_tick(timer_period);

    // Register callbacks
    log_debug("setting up mcpl");
    spin1_callback_on(MCPL_PACKET_RECEIVED, incoming_update_callback, 0);

    spin1_callback_on(TIMER_TICK, timer_callback, 2);
//    log_debug("MCPL callbacks requested for data input.");

    log_debug("Starting");
    simulation_run();
}
