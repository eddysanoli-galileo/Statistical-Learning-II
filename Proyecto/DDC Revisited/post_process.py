import numpy as np
from collections import defaultdict
from scipy import signal

# ======================================
# ADD TIMESTAMPS TO EVERY MEASURE
# ======================================

def add_measure_timestamps(song_tags):
    """Takes all the charts in a song's "note data" and goes through every measure.
    The "absolute time" (or the "timestamps") associated with every beat are obtained 
    and concatenated alongside with every other measure. The result is a numpy array 
    with as many columns as there were steps in a chart, plus an additional "timestamp"
    column. Each row in said matrix corresponds to a beat.
    """

    # Dict to store data for all charts
    chart_data = defaultdict(dict)  

    # For every chart in a song
    for chart in song_tags["notes"]:

        # All measures will be concatenated vertically
        # The initial empty vector has as many columns as each measure has columns
        measure_data = np.empty((0, chart["notedata"][0].shape[1]))

        # All timestamps (in seconds) for each beat 
        # will be concatenated vertically as well
        timestamp_data = np.empty((0, 1))

        # BPM for each beat
        bpm_data = np.empty((0,1))

        # =================
        # SEGMENT LENGTH
        # =================

        # BPMS for the song (applies for all difficulties)
        # They are converted into a numpy array
        BPMs = np.array(song_tags["bpms"])

        # If the song has no BPM
        if len(BPMs) <= 0:
            raise Exception("No value for the BPM attribute.")

        # If the song has more than 1 Beat,BPM pair
        if len(BPMs) > 1:

            # Seconds per beat.
            # 1 Beat/Min = 1 Beat / 60s -> 1 Second/Beat = 60s/1 Beat 
            SPBs = 60 / BPMs[:,1]

            # Difference between all subsequent "beat" timestamps
            # Length of a segment measured in beats.
            beat_diffs = np.diff(BPMs[:,0], axis=0)

            # Get the length of a "BPM segment" (segment in which the BPM
            # remains constant), measured in seconds. We dont take the last element
            # of the SPBs, as this is the last segment that ends until the end of the
            # song, it is not part of a segment with a measurable length.
            bpm_segment_len = beat_diffs * SPBs[0:-1]

            # Cumulative sum of bpm segment length
            cumulative_bpm_len = np.cumsum(bpm_segment_len)
            cumulative_bpm_len = np.append(0, cumulative_bpm_len)

        # =======================
        # BEATS TO ABSOLUTE TIME
        # =======================

        # For each measure in the note data
        # Remember: There are four beats in each measure.
        for measure_num, measure in enumerate(chart["notedata"]):

            # ===========================
            # BEAT INDEX
            # ===========================
            
            # Number of lines or rows in the current measure
            measure_len = measure.shape[0]

            # Each line is assigned a "line number":
            # A number from 1 to the length of the measure
            line_number = np.linspace(1, measure_len, measure_len)
            
            # If we were to assign an index to every row in all measures
            # we would have a "beat_index". This calculates the beat_index
            # of each line. 
            # 
            # Steps:
            # - Get the number of "beats passed" by getting the current measure
            #   number (because of the enumerate, the count goes from 0 to the number
            #   of the last measure) and multiplying it by the 4 beats in each measure.
            # - Add the "fractions of beat" that correspond to each of the lines of the
            #   current measure.
            beat_index = 4*measure_num + 4*(line_number / measure_len)

            # ===========================
            # BPM SEGMENT
            # ===========================
            
            # If there is more than 1 timestamp in "BPM"
            if len(BPMs) > 1:

                # Determine to which "bpm_segment" the current beats pertain. For this we
                # use the function "bisect" applied to the whole vector (searchsorted). This
                # function returns the index where a number should be placed in an array, to
                # keep the array sorted. 
                #
                # For example: 'searchsorted([1,2,3,4], 2.5)' will return 2 as using this index
                # to place the 2.5 inside the list, will place it after the 2.
                #
                # Given this, we can input our list of beat timestamps in "BPMs" as a first argument
                # and our list of beats as a second argument. This will give us the "next" beat segment
                # in which our beat should be placed to be greater than the previous beat timestamp.
                # By subtracting one from this result, we will get the beat segment each beat belongs to.
                # In short, we obtain the BPM segment to which every beat belongs to.
                bpm_segment_beat_idx = np.searchsorted(BPMs[:,0], beat_index, side="left") - 1

                # Cumulative bpm segment length for all current beats
                cumulative_bpm_beat = cumulative_bpm_len[bpm_segment_beat_idx]
            
            # If BPM only has one beat timestamp
            # - We asume that all beats pertain to the only segment available: 0
            # - There is nothing to "cumulative sum", so the sum is equal to 0
            elif len(BPMs) == 1:
                bpm_segment_beat_idx = np.zeros((measure_len)).astype(np.int)
                cumulative_bpm_beat = 0

            # If BPM has no timestamps, the stepfile is defective.
            else:
                raise Exception("No BPM provided for the song. Unable to process the file.")

            # =================
            # STOPS
            # =================

            # Stops for the current song
            stops = song_tags["stops"]

            # If stops is not empty
            if stops is not None:

                # Cumulative sum for all the error lengths (in seconds)
                cumulative_stop_len = np.cumsum(stops[:,1])

                # If the first BPM listed is not 0, we append a cero
                # at the beggining of the cumulative sum.
                if stops[0,0] != 0:
                    cumulative_stop_len = np.append(0, cumulative_stop_len)
                
                # To which "stop segment" each beat pertains to.
                # For reference return to the statement used to get "bpm_segment_per_beat"
                stop_segment_beat_idx = np.searchsorted(stops[:,0], beat_index, side="left")

                # Cumulative sum of stopped time for each beat
                cumulative_stop_beat = cumulative_stop_len[stop_segment_beat_idx]

            # If there are no stops, the cumulative per beat is 0
            else:
                cumulative_stop_beat = 0
            
            # =================
            # PARTIAL SEGMENT
            # =================

            # Steps:
            # 1. The BPM for each segment is converted to seconds per beat (SPB)
            # 2. We get the difference between the current beat and the las BPM segment limit (given in beats)
            # 3. The previous difference is converted into seconds by multiplying by the SPB (SPB x B = S)
            partial_segment_spb = (60 / BPMs[bpm_segment_beat_idx, 1])
            partial_segment = partial_segment_spb * (beat_index  - BPMs[bpm_segment_beat_idx, 0])

            # =================
            # OFFSET
            # =================

            # Offset and Stops for the song
            offset = song_tags["offset"]

            # Get absolute time
            # Get the time by doing the following:
            # - Add the duration of all previous BPM segments
            # - Add the duration of all previous stops
            # - Subtract the global offset
            # - Add the time elapsed from the last bpm segment and the current beat
            beat_abs_time = cumulative_bpm_beat + cumulative_stop_beat - offset + partial_segment

            # Timestamp and measure data is added to the existing data.
            timestamp   = np.reshape(beat_abs_time, (-1,1))
            measure_bpm = np.reshape(BPMs[bpm_segment_beat_idx,1], (-1,1))
            timestamp_data = np.vstack((timestamp_data, timestamp))
            measure_data   = np.vstack((measure_data, measure))
            bpm_data       = np.vstack((bpm_data, measure_bpm))

        # =======================
        # TIMESTAMP ADJUSTMENT
        # =======================

        # NOTE: This method returns a shifted array of timestamps for every beat. In other words,
        # if we were to have three beats (lets say beats 3, 4 and 5), their correspondence to the 
        # accompanying "timestamp" is NOT as follows
        #
        #                          Timestamps
        #   |Beat3|   |1000|        | 1.5s |
        #   |Beat4| = |0100| =/=>   | 2.3s |
        #   |Beat5|   |0010|        | 3.2s |
        #     ...       ..          | 4.1s |
        # 
        # Instead, we have to shift by one row the timestamps and attach a zero at the beggining
        # (deleting the end row in the process):
        #
        #                          Timestamps
        #   |Beat3|   |1000|        |  0s  |
        #   |Beat4| = |0100|  ==>   | 1.5s |
        #   |Beat5|   |0010|        | 2.3s |
        #     ...       ..          | 3.2s |
        #     ...       ..          | 4.1s |
        timestamp_data = np.vstack((0,timestamp_data[0:-1]))

        # =======================
        # DATA CONCATENATION
        # =======================

        # Timestamp and measure data is concatenated
        timed_measure_data = np.hstack((measure_data, timestamp_data, bpm_data))

        # We organize the "timed measure data" by putting it in a 
        # dict organized by chart type and difficulty.
        chart_data[chart["charttype"]][chart["difficulty"]] = timed_measure_data

    
    return chart_data


# Function to obtain the spectrogram of a song.
# Source: https://stackoverflow.com/questions/47954034/plotting-spectrogram-in-audio-analysis
def log_spectrogram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    """
    Calculate the spectrogram of a given "mono" audio.

    Params
    ------
        audio (array):
            Data read from a mono wav file.
        sample_rate (array):
            Original audio's sample rate
        window_size (float):
            Window size or size of frame in miliseconds.
        step_size (float):
            Stride or skip measured in miliseconds.
        eps (float):
            Small offset.
    """

    # Length of segment and number points to overlap 
    # See the following link for more info: 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    # The spectrogram is calculated using the previous data
    freqs, times, spec = signal.spectrogram(audio, fs=sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)

    # The log of the spectrogram is calculated
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)