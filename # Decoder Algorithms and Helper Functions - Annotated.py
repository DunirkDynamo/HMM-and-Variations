# Decoder Algorithms and Helper Functions with Enhanced Documentation

# Updated with detailed purpose, input, and output annotations.

def rejection_sampling(density_func, bounds, num_samples=1, args=[]):
    '''
    Purpose:
        Draws samples from a user-defined density function using rejection sampling. This method is useful for generating random samples from distributions where direct sampling is challenging.
    
    Input:
        - density_func [function] : A function that takes a sample point and returns the density at that point.
        - bounds       [tuple]   : A tuple (lower_bound, upper_bound) defining the range from which to sample.
        - num_samples  [int]     : The number of samples to draw (default is 1).
        - args         [list]    : Optional arguments to pass to the density function.
    
    Output:
        - np.array : An array of samples drawn from the distribution defined by the density function.
    '''
    # Determine the maximum value of the density function in the given bounds
    lower_bound, upper_bound = bounds
    sample_points = np.linspace(lower_bound, upper_bound, 1000)

    if args:
        max_density = max(density_func(x, *args) for x in sample_points)
    else:
        max_density = max(density_func(x) for x in sample_points)
    # List to store the accepted samples
    samples = []

    while len(samples) < num_samples:
        # Sample a candidate from the uniform distribution in the given range
        candidate = np.random.uniform(lower_bound, upper_bound)

        # Sample a uniform random value between 0 and max_density
        u = np.random.uniform(0, max_density)

        # Evaluate the density at the candidate point
        if args:
            density = density_func(candidate, *args)
        else:
            density = density_func(candidate)

        # Accept the candidate with probability proportional to its density
        if u < density:
            samples.append(candidate)

    return np.array(samples)

def inverse_gauss_phys(t, d, v, s):
    '''
    Purpose:
        Defines an inverse Gaussian probability density function, often used in physical simulations involving drift and volatility components.
    
    Input:
        - t [float] : Time parameter.
        - d [float] : Pore length or distance parameter.
        - v [float] : Classical drift speed in the medium.
        - s [float] : Volatility component scaling the noise parameter (units: distance/time^0.5).
    
    Output:
        - float : The density value calculated using the inverse Gaussian distribution.
    '''
    import numpy as np
    from math import pi as pi
    coef = d/s/np.sqrt(2*pi*t**3)
    arg  = (v*t - d)**2 / (2 * s**2 * t)

    density = coef * np.exp(-arg)

    return density

inverse_gauss_phys_vec = np.vectorize(inverse_gauss_phys)

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
########################################################### FUNCTIONS FOR GENERATING STEP SEQUENCE HMMS  #################################################################
########################################################### FUNCTIONS FOR GENERATING STEP SEQUENCE HMMS  #################################################################
########################################################### FUNCTIONS FOR GENERATING STEP SEQUENCE HMMS  #################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################   
def choose_random_state_index(states):
    '''
    Purpose:
        Selects a random state index from a list of states, assuming equal probability for each state unless specified otherwise.
    
    Input:
        - states [list] : A list of possible states.
    
    Output:
        - int : The index of the randomly chosen state.
    '''
    probabilities = [1 / len(states)] * len(states)  # Equal probabilities by default
    chosen_index = np.random.choice(len(states), p=probabilities)
    return chosen_index

def generate_state_sequence(states, transition_matrix, N):
    '''
    Purpose:
        Generates a sequence of states based on a transition matrix and initial random state selection.
    
    Input:
        - states            [list] : A list of possible states.
        - transition_matrix [2D array] : Matrix defining transition probabilities between states.
        - N                 [int] : The number of states to generate in the sequence.
    
    Output:
        - list : A sequence of state indices.
    '''
    current_state_index = choose_random_state_index(states)
    state_sequence = [current_state_index]

    for _ in range(1, N):
        next_state_index = np.random.choice(len(states), p=transition_matrix[current_state_index])
        state_sequence.append(next_state_index)
        current_state_index = next_state_index

    return state_sequence

def generate_durations_for_sequence(state_sequence, states):
    '''
    Purpose:
        Generates durations for each state in a given state sequence.
    
    Input:
        - state_sequence [list] : A list of state indices.
        - states         [list] : A list of state objects.
    
    Output:
        - list : A list of durations corresponding to each state in the sequence.
    '''
    durations = []

    for state_index in state_sequence:
        state = states[state_index]
        duration = state.duration_instance()
        durations.append(duration)

    return durations


def generate_sampling_times(durations, sample_rate):
    '''
    Purpose:
        Generates an array of sampling times based on state durations and the sampling rate.
    
    Input:
        - durations   [list] : A list of durations for each state.
        - sample_rate [float]: The rate at which samples are taken (Hz).
    
    Output:
        - np.array : An array of sampling times.
    '''
    cumulative_durations = np.cumsum(durations)
    dt = 1 / sample_rate
    sampling_times = []
    current_time = 0

    while current_time <= cumulative_durations[-1]:
        sampling_times.append(current_time)
        current_time += dt

    return np.array(sampling_times)

def get_durations_discretized(nobs, sample_rate, durations):
    '''
    Purpose:
        Converts continuous state durations into discretized intervals based on the number of observations and sample rate.
    
    Input:
        - nobs        [int]   : Total number of observations.
        - sample_rate [float] : Sampling rate in Hz.
        - durations   [list]  : A list of continuous durations for each state.
    
    Output:
        - np.array : An array of discretized durations.
    '''
    cudur = np.cumsum(durations)
    prevailing_state_index = 0
    t_sample = 0

    durations_discretized = np.zeros(len(durations))
    count = 0

    for i in range(nobs):
        if t_sample > cudur[prevailing_state_index]:
            durations_discretized[prevailing_state_index] = count
            prevailing_state_index += 1
            count = 1
        elif i == nobs - 1:
            count += 1
            durations_discretized[prevailing_state_index] = count
        else:
            count += 1

        t_sample += 1.0 / sample_rate

    return durations_discretized

def sample_from_sequence(sampling_times, state_sequence, states, durations):
    '''
    Purpose:
        Generates observations from a given state sequence and sampling times while identifying changepoint indices.
    
    Input:
        - sampling_times  [np.array] : An array of sampling times for the sequence.
        - state_sequence  [list]     : A sequence of state indices.
        - states          [list]     : A list of state objects.
        - durations       [list]     : A list of durations for each state in the sequence.
    
    Output:
        - np.array : Observations sampled from the sequence.
        - np.array : True states corresponding to each observation.
        - list     : Indices where state changes occur (changepoints).
    '''
    observations = []
    cumulative_durations = np.cumsum(durations)
    changepoint_indices = [0]
    current_state_index = 0

    true_states = []
    num_samples = len(sampling_times)
    for i in range(num_samples):
        sample_time = sampling_times[i]
        while current_state_index < len(cumulative_durations) and sample_time >= cumulative_durations[current_state_index]:
            current_state_index += 1
            changepoint_indices.append(i)

        if current_state_index >= len(state_sequence):
            break

        state_index = state_sequence[current_state_index]
        observation = states[state_index].draw()
        observations.append(observation)
        true_states.append(state_sequence[current_state_index])

    return np.array(observations), np.array(true_states), changepoint_indices


def plot_observations_with_true_signal(sampling_times, observations, state_sequence, states, durations):
    '''
    Purpose:
        Plots the observations against sampling times, overlaying the true signal corresponding to each state.

    Input:
        - sampling_times [np.array] : Array of sampling times.
        - observations   [np.array] : Observed data points.
        - state_sequence [list]     : Sequence of state indices.
        - states         [list]     : List of state objects.
        - durations      [list]     : Durations for each state in the sequence.

    Output:
        - None: The function generates a plot visualizing the observations and true signal.
    '''
    plt.figure(figsize=(12, 6))

    # Scatter plot of observations
    plt.scatter(sampling_times[:len(observations)], observations, color='blue', label='Observations', s=10)
    plt.plot(sampling_times[:len(observations)], observations, lw=1)

    # Define colors for each state
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33A1']

    # Plot the true signal
    cumulative_durations = np.cumsum(durations)
    for i, state_index in enumerate(state_sequence):
        mean = states[state_index].mean
        start_x = 0 if i == 0 else cumulative_durations[i - 1]
        duration = durations[i]
        plt.hlines(mean, start_x, start_x + duration, color=colors[i % len(colors)], linewidth=2, label=f'State {i + 1}')

    # Set labels and title
    plt.xlabel('Sampling Times (s)')
    plt.ylabel('Observations')
    plt.title('Observations vs. Sampling Times with True Signal')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid()
    plt.show()

def split_time_series(time_series, changepoints):
    '''
    Purpose:
        Splits a time series into segments based on provided changepoint indices.

    Input:
        - time_series  [np.ndarray] : The input time series data.
        - changepoints [list]       : Indices indicating where to split the time series.

    Output:
        - list: A list of segments (np.ndarray), each representing a portion of the time series.
    '''
    if not isinstance(time_series, np.ndarray):
        raise TypeError("time_series must be a numpy array")
    if not isinstance(changepoints, (list, np.ndarray)):
        raise TypeError("changepoints must be a list or numpy array")
    if any(cp < 0 or cp >= len(time_series) for cp in changepoints):
        raise ValueError("Changepoints must be within the range of the time series")

    # Ensure changepoints are sorted
    changepoints = sorted(changepoints)

    # Add the end of the series to the changepoints for final split
    changepoints.append(len(time_series))

    # Generate sub-arrays
    segments = [
        time_series[changepoints[i]:changepoints[i + 1]].ravel()
        for i in range(len(changepoints) - 1)
    ]
    return segments


def chunks_to_pointwise(observations, L, most_probable_chunks):
    '''
    Purpose:
        Converts chunk-based data into a pointwise sequence, associating each observation with its most probable chunk.

    Input:
        - observations          [np.array] : Array of observed data points.
        - L                     [int]      : Number of points per chunk.
        - most_probable_chunks  [list]     : List of most probable chunks corresponding to each observation.

    Output:
        - np.array : Pointwise sequence of chunk assignments for each observation.
    '''
    from math import floor
    num_obs = len(observations)
    pointwise = np.zeros(num_obs, dtype=int)
    num_chunks = floor(num_obs / L)
    num_leftover = int(num_obs - num_chunks * L)

    for i in range(num_chunks):
        start_index = i * L
        end_index = (i + 1) * L
        pointwise[start_index:end_index] = most_probable_chunks[i]

    if num_leftover > 0:
        emission_index = num_chunks * L
        for i in range(num_leftover):
            pointwise[emission_index] = most_probable_chunks[num_chunks + i]
            emission_index += 1

    return pointwise

def create_dataset(sample_rate, means, variance, d, dvs_params, num_steps=7):
    '''
    Purpose:
        Creates a synthetic dataset using step-sequence HMMs with defined parameters for states and sampling.

    Input:
        - sample_rate [float] : Sampling rate in Hz.
        - means       [list]  : List of mean values for each state.
        - variance    [list]  : List of variances corresponding to each state.
        - d           [float] : Distance parameter used in the duration distribution.
        - dvs_params  [list]  : Parameters for the duration distribution of each state.
        - num_steps   [int]   : Number of states in the sequence (default is 7).

    Output:
        - tuple: Contains the following elements:
            - observations         [np.array] : Generated observations.
            - true_states          [np.array] : Ground-truth state indices for each observation.
            - durations            [list]     : List of durations for each state.
            - changepoint_indices  [list]     : Indices where state transitions occur.
            - states               [list]     : List of state objects.
            - transition_matrix    [2D array] : Transition matrix used for state generation.
            - cdf_lookup_tables    [list]     : Precomputed cumulative distribution lookup tables.
            - state_sequence       [list]     : Sequence of state indices.
    '''
    import numpy as np

    # Create instances of State
    states = [State(mean, var, inverse_gauss_phys, dvs) for mean, var, dvs in zip(means, variance, dvs_params)]

    num_states = len(states)
    transition_matrix = np.random.rand(num_states, num_states)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    state_sequence = generate_state_sequence(states, transition_matrix, num_steps)
    durations = generate_durations_for_sequence(state_sequence, states)
    cumulative_duration = np.cumsum(durations)

    sampling_times = generate_sampling_times(durations, sample_rate)
    durations_discretized = get_durations_discretized(len(sampling_times), sample_rate, durations)

    observations = np.concatenate([
        states[state_sequence[i]].draw(num_samples=int(durations_discretized[i])) for i in range(len(durations_discretized))
    ])
    true_states = np.concatenate([
        np.repeat(state_sequence[i], int(durations_discretized[i])) for i in range(len(durations_discretized))
    ])
    changepoint_indices = np.cumsum(durations_discretized)

    cdf_lookup_tables = []
    for s in states:
        dmax = get_ROI(states[0], 0.0, sample_rate, alpha_low=0.1, alpha_upp=0.9999)[1]
        cdf_lookup_tables.append(get_cumulative_lookup(s.duration_dist, 0, sample_rate, dmax, s.duration_dist_params))

    return (
        observations, true_states, durations, changepoint_indices, states,
        transition_matrix, cdf_lookup_tables, state_sequence
    )

def create_dataset_2(states, transition_matrix, sample_rate, num_steps):
    '''
    Purpose:
        Generates a synthetic dataset with an alternative approach, using pre-defined states and transition matrix.

    Input:
        - states            [list]     : List of state objects.
        - transition_matrix [2D array] : Predefined matrix for state transitions.
        - sample_rate       [float]    : Sampling rate in Hz.
        - num_steps         [int]      : Number of states in the sequence.

    Output:
        - tuple: Contains the following elements:
            - observations         [np.array] : Generated observations.
            - state_sequence       [list]     : Sequence of state indices.
            - changepoint_indices  [list]     : Indices where state transitions occur.
            - durations            [list]     : List of durations for each state.
    '''
    random_state_index = choose_random_state_index(states)
    state_sequence = generate_state_sequence(states, transition_matrix, num_steps)
    durations = generate_durations_for_sequence(state_sequence, states)
    sampling_times = generate_sampling_times(durations, sample_rate)

    observations, true_states, changepoint_indices = sample_from_sequence(
        sampling_times, state_sequence, states, durations
    )

    return observations, state_sequence, changepoint_indices, durations

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
########################################################### CLASSES AND FUNCTION FOR DEFINING HMMS  ######################################################################
########################################################### CLASSES AND FUNCTION FOR DEFINING HMMS  ######################################################################
########################################################### CLASSES AND FUNCTION FOR DEFINING HMMS  #######################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
class State:
    '''
    Purpose:
        Represents a single state in an HMM or HSMM, with associated emission and duration distributions.

    Input:
        - mean                 [float] : Mean of the emission distribution.
        - variance             [float] : Variance of the emission distribution.
        - duration_dist        [function] : Probability density function for state durations.
        - duration_dist_params [tuple] : Parameters for the duration distribution.
    '''
    def __init__(self, mean, variance, duration_dist, duration_dist_params):
        # For the normally distributed current samples:
        self.mean                 = mean
        self.variance             = variance 
        # For the distribution of step durations:
        self.duration_dist        = duration_dist
        self.duration_dist_params = duration_dist_params

    def prob_n(self, n, sample_rate):
        '''
        Purpose:
            Computes the probability of a given duration `n` for this state.

        Input:
            - n           [int]   : Duration to compute the probability for.
            - sample_rate [float] : Sampling rate in Hz.

        Output:
            - float : The computed probability for the duration `n`.
        '''
        from scipy.integrate import quad

        t1 = quad(self.duration_dist, n / sample_rate, (n + 1) / sample_rate, tuple(self.duration_dist_params))[0]
        t2 = quad(self.duration_dist, (n - 1) / sample_rate, n / sample_rate, tuple(self.duration_dist_params))[0]
        J  = (n + 1) * t1 - (n - 1) * t2

        def expectation_duration_dist(x):
            return x * self.duration_dist(x, *self.duration_dist_params)

        t3 = quad(expectation_duration_dist, (n - 1) / sample_rate, n / sample_rate)[0]
        t4 = quad(expectation_duration_dist, n / sample_rate, (n + 1) / sample_rate)[0]
        W  = (t3 - t4) * sample_rate

        output = J + W
        return output

    def joint_n_xvec(self, x, gamma, tol):
        '''
        Purpose:
            Computes the joint probability of a given duration `n` and observation vector `x`.

        Input:
            - x     [np.array] : Observation vector.
            - gamma [float]    : Time scaling factor.
            - tol   [float]    : Tolerance for numerical calculations.

        Output:
            - float : Joint probability of `n` and `x`.
        '''
        from math import pi

        n = len(x)
        delta = x - self.mean
        vhat  = np.dot(delta, delta) / n
        A     = 1. / (2 * pi * self.variance) ** (n / 2.)
        xdensity = A * np.exp(-0.5 * n * vhat / self.variance)

        ndensity = self.prob_n(n, 1. / gamma)
        joint_density = ndensity * xdensity
        return joint_density

    def log_joint_n_xvec(self, x, gamma, tol):
        '''
        Purpose:
            Computes the logarithm of the joint probability for `n` and `x` for numerical stability.

        Input:
            - x     [np.array] : Observation vector.
            - gamma [float]    : Time scaling factor.
            - tol   [float]    : Tolerance for numerical calculations.

        Output:
            - float : Logarithm of the joint probability of `n` and `x`.
        '''
        from math import pi

        n = len(x)
        delta = x - self.mean
        vhat  = np.dot(delta, delta) / n
        log_A = -n * np.log(2 * pi * self.variance) / 2
        log_xdensity = log_A - n * vhat / (2 * self.variance)

        ndensity = self.prob_n(n, 1. / gamma)
        log_joint_density = np.log(ndensity) + log_xdensity
        return log_joint_density

    def draw(self, num_samples=1):
        '''
        Purpose:
            Generates random samples from the state's emission distribution.

        Input:
            - num_samples [int] : Number of samples to generate (default is 1).

        Output:
            - np.array : Array of generated samples.
        '''
        return np.random.normal(loc=self.mean, scale=np.sqrt(self.variance), size=num_samples)

    def duration_instance(self):
        '''
        Purpose:
            Draws a single instance of a duration from the state's duration distribution.

        Input:
            - None

        Output:
            - float : A randomly drawn duration value.
        '''
        return rejection_sampling(self.duration_dist, bounds=(0.000001, 10), num_samples=1, args=self.duration_dist_params)[0]

    def emission_probability(self, obs):
        '''
        Purpose:
            Calculates the Gaussian emission probability for a given observation.

        Input:
            - obs [float] : A single observed value.

        Output:
            - float : Emission probability of the observation.
        '''
        return (1 / (np.sqrt(self.variance * 2 * np.pi))) * np.exp(-0.5 * ((obs - self.mean) / np.sqrt(self.variance)) ** 2)



def get_cumulative_probability(pdf, low_bound, upp_bound, args):
    '''
    Purpose:
        Computes the cumulative probability of a given probability density function (PDF) over a specified range.

    Input:
        - pdf       [function] : The probability density function.
        - low_bound [float]    : Lower bound of the integration range.
        - upp_bound [float]    : Upper bound of the integration range.
        - args      [tuple]    : Additional arguments to pass to the PDF function.

    Output:
        - float : The cumulative probability over the specified range.
    '''
    from scipy.integrate import quad
    if args:
        result, error = quad(pdf, low_bound, upp_bound, tuple(args))
    else:
        result, error = quad(pdf, low_bound, upp_bound)
    return result

def get_ROI(S, thresh, sample_rate, alpha_low=0.1, alpha_upp=0.9):
    '''
    Purpose:
        Determines the range of interest (ROI) for a state's duration distribution, based on given thresholds.

    Input:
        - S            [State]  : The state object containing the duration distribution.
        - thresh       [float]  : Threshold for the integration.
        - sample_rate  [float]  : Sampling rate in Hz.
        - alpha_low    [float]  : Lower quantile for the ROI (default is 0.1).
        - alpha_upp    [float]  : Upper quantile for the ROI (default is 0.9).

    Output:
        - list : A two-element list [nlow, nupp] defining the lower and upper bounds of the ROI.
    '''
    import numpy as np
    from math import floor, ceil
    from scipy.optimize import fsolve

    x0 = S.duration_dist_params[0] / S.duration_dist_params[1]
    low_cut = fsolve(lambda tcut: get_cumulative_probability(S.duration_dist, thresh, tcut, S.duration_dist_params) - alpha_low, x0=x0)
    upp_cut = fsolve(lambda tcut: get_cumulative_probability(S.duration_dist, thresh, tcut, S.duration_dist_params) - alpha_upp, x0=x0)

    nlow = floor(low_cut * sample_rate)
    nupp = ceil(upp_cut * sample_rate)

    return [nlow, nupp]

def get_cumulative_lookup(pdf, min_bound, sample_rate, dmax, args):
    '''
    Purpose:
        Creates a lookup table for cumulative probabilities of a given PDF, sampled at regular intervals.

    Input:
        - pdf          [function] : The probability density function.
        - min_bound    [float]    : The lower bound for the PDF.
        - sample_rate  [float]    : Sampling rate in Hz.
        - dmax         [int]      : Maximum duration to consider.
        - args         [tuple]    : Additional arguments to pass to the PDF function.

    Output:
        - np.array : Lookup table containing cumulative probabilities.
    '''
    lookup_table = np.zeros(dmax)

    lookup_table = np.array([
        get_cumulative_probability(pdf, min_bound, d / sample_rate, args)
        for d in (np.arange(dmax) + 1)
    ])

    return lookup_table


##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
########################################################### DECODER ALGORITHMS AND HELPER FUNCTIONS ######################################################################
########################################################### DECODER ALGORITHMS AND HELPER FUNCTIONS ######################################################################
########################################################### DECODER ALGORITHMS AND HELPER FUNCTIONS ######################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
def log_viterbi_algorithm(obs, states, start_prob, transition_matrix):    
    # Step 1: Initialize Variables
    num_states        = len(states)
    num_obs           = len(obs)
    log_viterbi_table = [[0.0 for _ in range(num_states)] for _ in range(num_obs)]
    backpointer       = [[0 for _ in range(num_states)] for _ in range(num_obs)]

    # Step 2: Calculate Probabilities
    for t in range(num_obs):
        for s in range(num_states):
            if t == 0:
                log_viterbi_table[t][s] = np.log(start_prob[s]) + np.log(states[s].emission_probability(obs[t]))
            else:
                log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                log_viterbi_table[t][s] = log_max_prob + np.log(states[s].emission_probability(obs[t]))
                backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]))

    # Step 3: Traceback and Find Best Path
    best_path_prob    = max(log_viterbi_table[-1])
    best_path_pointer = log_viterbi_table[-1].index(best_path_prob)
    #best_path_pointer = max(range(num_states), key=lambda s: viterbi_table[-1][s])
    best_path         = [best_path_pointer]
    for t in range(len(obs)-1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    # Step 4: Return Best Path and other info
    return best_path, backpointer, log_viterbi_table

    
def viterbi_algorithm_vector_new(obs, states, start_prob, transition_matrix, gamma, tol=1e-8):
    # Step 2: Initialize Variables
    num_states        = len(states)
    num_obs           = len(obs)
    log_viterbi_table = [[0.0 for _ in range(num_states)] for _ in range(num_obs)]
    backpointer       = [[0 for _ in range(num_states)] for _ in range(num_obs)]

    # Step 3: Calculate Probabilities
    for t in range(num_obs):
        for s in range(num_states):
            if t == 0:
                log_viterbi_table[t][s] = np.log(start_prob[s]) + states[s].log_joint_n_xvec(obs[t], gamma, tol=tol)
            else:
                log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                log_viterbi_table[t][s] = log_max_prob + states[s].log_joint_n_xvec(obs[t], gamma, tol=tol)
                backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] +  np.log(transition_matrix[prev_s][s]))

    # Step 4: Traceback and Find Best Path
    best_path_prob    = max(log_viterbi_table[-1])
    best_path_pointer = max(range(num_states), key=lambda s: log_viterbi_table[-1][s])
    best_path         = [best_path_pointer]
    for t in range(len(obs)-1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    # Step 5: Return Best Path
    return best_path, backpointer, log_viterbi_table

def log_viterbi_algo_chunks(obs, states, start_prob, transition_matrix, L):
    from math import floor
    # Step 1: Initialize Variables
    num_states        = len(states)
    num_obs           = len(obs)
    num_chunks        = floor(num_obs/L)
    num_leftover      = int(num_obs - L * num_chunks) 
    log_viterbi_table = [[0.0 for _ in range(num_states)] for _ in range(num_chunks+num_leftover)]
    backpointer       = [[0 for _ in range(num_states)] for _ in range(num_chunks+num_leftover)]

    # Step 2: Calculate Probabilities
    
    for t in range(num_chunks):
        start_index = t * L
        end_index   = start_index + L
        for s in range(num_states):
            if t == 0:
                log_viterbi_table[t][s] = np.log(start_prob[s]) + np.sum(np.array([np.log(states[s].emission_probability(obs[i])) for i in range(start_index, end_index)]))
            else:
                log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                log_viterbi_table[t][s] = log_max_prob + np.sum(np.array([np.log(states[s].emission_probability(obs[i])) for i in range(start_index, end_index)]))
                backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]))

    if num_leftover > 0:
        emission_index = num_chunks*L 
        for t in range(num_chunks, num_chunks + num_leftover):
            for s in range(num_states):
                log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                log_viterbi_table[t][s] = log_max_prob + np.log(states[s].emission_probability(obs[emission_index]))
                backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]))
            emission_index+=1
    # Step 3: Traceback and Find Best Path
    best_path_prob    = max(log_viterbi_table[-1])
    best_path_pointer = log_viterbi_table[-1].index(best_path_prob)
    #best_path_pointer = max(range(num_states), key=lambda s: viterbi_table[-1][s])
    best_path         = [best_path_pointer]
    for t in range(num_chunks+num_leftover-1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    # Step 4: Return Best Path and other info
    return best_path, backpointer, log_viterbi_table

def log_sum_exp(log_probs):
    """
    Computes the log of the sum of exponentials of input log probabilities.
    This avoids underflow/overflow issues with small probabilities.
    """
    max_log = max(log_probs)
    return max_log + math.log(sum(math.exp(lp - max_log) for lp in log_probs))


def forward_backward_log(obs, states, start_log_prob, transition_log_matrix):
    # Step 1: Initialize Variables
    num_states         = len(states)
    num_obs            = len(obs)
    forward_log_table  = [[-math.inf for _ in range(num_states)] for _ in range(num_obs)]
    backward_log_table = [[-math.inf for _ in range(num_states)] for _ in range(num_obs)]

    # Step 2: Forward Pass (Initialization)
    for s in range(num_states):
        forward_log_table[0][s] = start_log_prob[s] + math.log(states[s].emission_probability(obs[0]))

    # Step 3: Forward Pass (Recursion)
    for t in range(1, num_obs):
        for s in range(num_states):
            log_probs = [
                forward_log_table[t - 1][prev_s] + transition_log_matrix[prev_s][s]
                for prev_s in range(num_states)
            ]
            forward_log_table[t][s] = log_sum_exp(log_probs) + math.log(states[s].emission_probability(obs[t]))

    # Step 4: Backward Pass (Initialization)
    for s in range(num_states):
        backward_log_table[-1][s] = 0.0  # log(1) = 0

    # Step 5: Backward Pass (Recursion)
    for t in range(num_obs - 2, -1, -1):
        for s in range(num_states):
            log_probs = [
                backward_log_table[t + 1][next_s]
                + transition_log_matrix[s][next_s]
                + math.log(states[next_s].emission_probability(obs[t + 1]))
                for next_s in range(num_states)
            ]
            backward_log_table[t][s] = log_sum_exp(log_probs)

    # Step 6: Posterior Probabilities
    posterior_log_probs = [
        [forward_log_table[t][s] + backward_log_table[t][s] for s in range(num_states)]
        for t in range(num_obs)
    ]

    # Normalize posterior probabilities in log-space
    posterior_probs = []
    for t in range(num_obs):
        log_norm_factor = log_sum_exp(posterior_log_probs[t])
        posterior_probs.append([math.exp(lp - log_norm_factor) for lp in posterior_log_probs[t]])

    # Step 7: Most Likely State at Each Time Step
    most_likely_states = [max(range(num_states), key=lambda s: posterior_probs[t][s]) for t in range(num_obs)]

    return posterior_probs, most_likely_states, forward_log_table, backward_log_table





import numpy as np
def log_sum_exp_logs(log_a, log_b):
    """
    Compute log(exp(log_a) + exp(log_b)) in a numerically stable way.
    """
    M = max(log_a, log_b)
    return M + np.log(np.exp(log_a - M) + np.exp(log_b - M))


def JHSMM_weak(observations, states, start_prob, trans_prob, sample_rate, cdf_table):
    """
    Viterbi algorithm with tracking of consecutive state occurrences.

    Parameters:
        observations: list of observations
        states: list of possible states (objects with an emission_probability method)
        start_prob: dict of starting probabilities for each state
        trans_prob: 2D list, transition probabilities between states (indexed by state indices)

    Returns:
        tuple:
            - list of the most likely states (optimal path)
            - float, the log probability of the optimal path
            - 2D list of running counts of consecutive occurrences for each state at each time step
    """
    n_obs      = len(observations)
    num_states = len(states)
    
    # Initialize dynamic programming tables
    viterbi_probs      = np.full((n_obs, num_states), -np.inf)  # Log probabilities of paths
    back_pointers      = np.zeros((n_obs, num_states), dtype=int)  # Backtracking table
    consecutive_counts = np.zeros((n_obs, num_states), dtype=int)  # Running counts of consecutive occurrences
    
    # Initialization step (t = 0)
    for s in range(num_states):
        viterbi_probs[0, s]      = np.log(start_prob[s]) + np.log(states[s].emission_probability(observations[0]))
        back_pointers[0, s]      = -1  # No previous state at t=0
        consecutive_counts[0, s] = 1  # First occurrence of each state

    # Iteration step (t = 1 to n_obs - 1)
    for t in range(1, n_obs):
        for s in range(num_states):  # Current state
            max_prob               = -np.inf
            best_prev_state        = -1
            
            for prev_s in range(num_states):  # Previous state
                d_prevailing = consecutive_counts[t-1][prev_s]
                if s != prev_s:
                    if d_prevailing >= len(cdf_table[prev_s]):
                        duration_factor = 1
                    else:
                        duration_factor = cdf_table[prev_s][d_prevailing]#get_cumulative_probability(states[prev_s].duration_dist, 0.00001, consecutive_counts[t-1, prev_s]/sample_rate, states[prev_s].duration_dist_params)
                    prob = viterbi_probs[t-1, prev_s] + np.log(trans_prob[prev_s][s]) + np.log(duration_factor)
                else:
                    prob          = viterbi_probs[t-1, prev_s] + np.log(trans_prob[prev_s][s]) 
                
                if prob > max_prob:
                    max_prob        = prob
                    best_prev_state = prev_s

             # Update running count based on the best previous state
            if best_prev_state == s:
                best_consecutive_count = consecutive_counts[t-1, best_prev_state] + 1
            else:
                best_consecutive_count = 1
            # Update tables
            viterbi_probs[t, s]      = max_prob + np.log(states[s].emission_probability(observations[t]))
            back_pointers[t, s]      = best_prev_state
            consecutive_counts[t, s] = best_consecutive_count # Default to 1 if no consecutive match

    # Termination step: Find the most likely final state
    last_state = np.argmax(viterbi_probs[-1])
    log_prob   = viterbi_probs[-1, last_state]
    
    # Backtrack to find the optimal path
    optimal_path = [last_state]
    for t in range(n_obs - 1, 0, -1):
        last_state = back_pointers[t, last_state]
        optimal_path.append(last_state)
    
    optimal_path.reverse()  # Reverse to get the path from start to end
    return optimal_path, log_prob, consecutive_counts




def JHSMM_strong(observations, states, start_prob, trans_prob, sample_rate, cdf_table):
    """
    Viterbi algorithm with tracking of consecutive state occurrences.

    Parameters:
        observations: list of observations
        states: list of possible states (objects with an emission_probability method)
        start_prob: dict of starting probabilities for each state
        trans_prob: 2D list, transition probabilities between states (indexed by state indices)

    Returns:
        tuple:
            - list of the most likely states (optimal path)
            - float, the log probability of the optimal path
            - 2D list of running counts of consecutive occurrences for each state at each time step
    """
    n_obs      = len(observations)
    num_states = len(states)
    
    # Initialize dynamic programming tables
    viterbi_probs      = np.full((n_obs, num_states), -np.inf)  # Log probabilities of paths
    back_pointers      = np.zeros((n_obs, num_states), dtype=int)  # Backtracking table
    consecutive_counts = np.zeros((n_obs, num_states), dtype=int)  # Running counts of consecutive occurrences
    
    # Initialization step (t = 0)
    for s in range(num_states):
        viterbi_probs[0, s]      = np.log(start_prob[s]) + np.log(states[s].emission_probability(observations[0]))
        back_pointers[0, s]      = -1  # No previous state at t=0
        consecutive_counts[0, s] = 1  # First occurrence of each state

    # Iteration step (t = 1 to n_obs - 1)
    for t in range(1, n_obs):
        for s in range(num_states):  # Current state
            max_prob               = -np.inf
            best_prev_state        = -1
            
            for prev_s in range(num_states):  # Previous state
                d_prevailing = consecutive_counts[t-1][prev_s]
                if s != prev_s:
                    if d_prevailing >= len(cdf_table[prev_s]):
                        duration_factor = 1
                    else:
                        duration_factor = cdf_table[prev_s][d_prevailing]#get_cumulative_probability(states[prev_s].duration_dist, 0.00001, consecutive_counts[t-1, prev_s]/sample_rate, states[prev_s].duration_dist_params)
                    prob = viterbi_probs[t-1, prev_s] + np.log(trans_prob[prev_s][s]) + np.log(duration_factor)
                else:
                    if d_prevailing < len(cdf_table[prev_s]):
                        log1            = np.log(1. - trans_prob[prev_s][s]) + np.log(cdf_table[prev_s][d_prevailing])
                        update_factor   = 1. - np.exp(log1)
                        prob            = viterbi_probs[t-1, prev_s] + np.log(update_factor) #viterbi_probs[t-1, prev_s] + logsumexp_out
                    else:
                        update_factor   = trans_prob[prev_s][s] 
                        prob            = viterbi_probs[t-1, prev_s] + np.log(update_factor) #viterbi_probs[t-1, prev_s] + logsumexp_out
                if prob > max_prob:
                    max_prob        = prob
                    best_prev_state = prev_s

             # Update running count based on the best previous state
            if best_prev_state == s:
                best_consecutive_count = consecutive_counts[t-1, best_prev_state] + 1
            else:
                best_consecutive_count = 1
            # Update tables
            viterbi_probs[t, s]      = max_prob + np.log(states[s].emission_probability(observations[t]))
            back_pointers[t, s]      = best_prev_state
            consecutive_counts[t, s] = best_consecutive_count # Default to 1 if no consecutive match

    # Termination step: Find the most likely final state
    last_state = np.argmax(viterbi_probs[-1])
    log_prob   = viterbi_probs[-1, last_state]
    
    # Backtrack to find the optimal path
    optimal_path = [last_state]
    for t in range(n_obs - 1, 0, -1):
        last_state = back_pointers[t, last_state]
        optimal_path.append(last_state)
    
    optimal_path.reverse()  # Reverse to get the path from start to end
    return optimal_path, log_prob, consecutive_counts



def viterbi_hsmm_recurdur_fly_2(obs, states, start_prob, transition_matrix, min_duration, max_duration, sample_rate, cdf_lookup):
    def get_best_duration_index(valid_states, best_duration):
        num_valid_states = len(valid_states)
        for i in range(num_valid_states):
            if best_duration == valid_states[i]:
                print('Best Index: {}'.format(i))
                return i
        
        return None


    num_states = len(states)
    num_obs    = len(obs)

    #min_duration = np.repeat(2, len(states))
    # Log-probability tables
    log_viterbi_table = []
    valid_state_table = []
    backpointer       = [] # [[np.array([None, None]) for _ in range(num_states)] for _ in range(num_obs)]  # Stores [prev_state, duration]

    # Log of the transition matrix:
    log_transition_matrix = np.log(transition_matrix)

    # Recurrence (includes initialization logic for t = 0)
    for t in range(num_obs):
        log_viterbi_table.append([])
        valid_state_table.append([])
        backpointer.append([])
        for s in range(num_states):
            log_viterbi_table[t].append([])
            valid_state_table[t].append([])
            backpointer[t].append([])
            # Initialize log of the joint emission probability for recursive calculation. We pre-calculate up to the minimum duration minus one.
            # Why minus one? Because it allows us to add a single new emission to the sum as we loop over valid durations beginning at the minimum
            # duration. If we did not include the "minus one", then the minimum duration would be a special case and we would require an if-statement
            # to check the value of duration at each iteration in the loop. 
            
                
            
            if t > 0:
                log_emission_prob  = np.log(states[s].emission_probability(obs[t])) 
                log_emission_prob  += np.sum(np.log(states[s].emission_probability(obs[t-min_duration[s]+2:t]))) # Note that this will "wrap" if min_duration[s] > t... it doesn't matter though because, in such a scenario, we break the loop further ahead
            else:
                log_emission_prob = 0
            
            # Loop over possible values of duration, d. Note that d being within a state's maximum and minimum values does not guarantee
            # that it is a valid duration. It must also either be the first state in the sequence or have a valid prior state. To have
            # a valid prior state, the duration must be such that it leaves a sufficient number of prior points, where the number of prior 
            # points is sufficient if it is greater than or equal to the minimum duration of at least one state class.
            for d_index in range(max_duration[s] - min_duration[s] + 1):
                d = d_index + min_duration[s]
                if t - (d - 1) < 0:  # Ensure valid duration
                    break

                log_emission_prob += np.log(states[s].emission_probability(obs[t-d+1]))
                log_duration_prob  = np.log(states[s].prob_n(d, sample_rate))
                if t - (d - 1) == 0:  # Initialization case
                    # Calculate emission probability recursively      
                    if t == 0:
                        log_emission_prob  = np.log(states[s].emission_probability(obs[t]))  
                    else:
                        log_emission_prob  = np.log(states[s].emission_probability(obs[t])) 
                        log_emission_prob  += np.sum(np.log(states[s].emission_probability(obs[:t])))
                    log_prob           = np.log(start_prob[s]) + log_duration_prob + log_emission_prob
                    log_viterbi_table[t][s].append(log_prob)
                    valid_state_table[t][s].append(d)
                    backpointer[t][s].append(np.array([None, None]))
                else:  # General case where t-(d-1) > 0 --> that is, we are looking for valid states that are not the first state in the sequence
                    # List of transition probabilities along with prev_s and prev_d
                    transition_probs = []
                    for prev_s in range(num_states):
                        if valid_state_table[t-d][prev_s]:
                            #if t-(d-1) >= min(valid_state_table[t-d][prev_s]):#(min_duration[prev_s]-1):
                            for prev_d_index in range(len(valid_state_table[t-d][prev_s])):
                                        
                                # Calculate emission probability recursively
                                prev_d = valid_state_table[t-d][prev_s][prev_d_index]
                                transition_probs.append( (np.array(log_viterbi_table[t - d][prev_s][prev_d_index]) + log_transition_matrix[prev_s][s] + np.log(cdf_lookup[prev_s][prev_d]) , prev_s, prev_d))

                    if transition_probs:  # Check if transitions exist
                        # Find the index of the maximum transition probability
                        max_tuple           = max(transition_probs, key=lambda x: x[0])
                        max_transition_prob = max_tuple[0]
                        prev_s              = max_tuple[1] 
                        prev_d              = max_tuple[2]
                        
                        # Update log-probability table and valid state table:
                        log_prob           = max_transition_prob  + log_emission_prob + log_duration_prob #+ np.log(1.-cdf_lookup[s][d]) 
                        log_viterbi_table[t][s].append(log_prob)
                        valid_state_table[t][s].append(d)
                        backpointer[t][s].append(np.array([prev_s, prev_d]))

                
               

    # Traceback
    best_path = []
    best_state, best_duration_index = max(
                                    ((s, d_index) for s in range(num_states) for d_index in range(len(log_viterbi_table[-1][s]))),
                                    key=lambda x: log_viterbi_table[-1][x[0]][x[1]]
                                    )

    best_duration = valid_state_table[-1][best_state][best_duration_index]#best_duration_index + min_duration[best_state]
    best_path.append((best_state, best_duration))

    t                = num_obs - 1
    current_duration = best_duration
    while t - current_duration >= 0:
        #best_duration_index = get_best_duration_index(valid_state_table[t-best_duration])
        prev_state, prev_duration = backpointer[t][best_state][best_duration_index]

        best_path.insert(0, (prev_state, prev_duration))
        prev_duration_index = get_best_duration_index(valid_state_table[t-current_duration][prev_state], prev_duration)
        
        t                  -= current_duration
        current_duration    = prev_duration
        best_state          = prev_state
        best_duration_index = prev_duration_index
        
        if current_duration == None:
            break
    return best_path, backpointer, log_viterbi_table, valid_state_table