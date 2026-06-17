### Utility functions for processing intrinsic ephys using IPFX

import numpy as np
from ipfx.data_set_features import extractors_for_sweeps
from ipfx.stimulus_protocol_analysis import LongSquareAnalysis
from ipfx.sweep import Sweep, SweepSet
from ipfx.error import FeatureError
try:
    from ipfx.chirp_features import extract_chirp_fft
except ImportError:
    from ipfx.chirp import chirp_amp_phase
    def extract_chirp_fft(sweep_set, min_freq=1, max_freq=15, end=20.6):
        resistance, reactance, freq = chirp_amp_phase(sweep_set, min_freq=min_freq, max_freq=max_freq, end=end)
        if resistance is None:
            return {}
        peak_idx = np.argmax(resistance)
        baseline = resistance.mean()
        return {
            'peak_ratio': float(resistance[peak_idx] / baseline) if baseline != 0 else 0.0,
            'peak_freq': float(freq[peak_idx]),
            'z_max': float(resistance.max()),
            'z_min': float(resistance.min()),
            'z_mean': float(baseline),
            'phase_at_peak': float(reactance[peak_idx]),
        }

import pandas as pd

def get_complete_long_square_features(analysis):
    """Extract the full feature set from a raw LongSquareAnalysis result dict.

    Works directly on the pandas objects returned by lsa.analyze() — no
    lsa.as_dict() call needed.  Extracts:
      • Scalar subthreshold features  (sag, IR, tau, fi_fit_slope)
      • Per-sweep ISI / rate features  (hero & rheobase sweeps)
      • First-spike AP waveform        (threshold_v, upstroke, downstroke,
                                        upstroke_downstroke_ratio, width,
                                        fast_trough_v, peak_v, peak_deltav,
                                        fast_trough_deltav, adp_v)
      • Cross-sweep adaptation         (adapt_mean, isi_cv_mean,
                                        *_adapt_ratio comparing last vs first
                                        spiking sweep)
    """
    out = {}

    # ── Scalar top-level features ────────────────────────────────────────────
    for k in ('rheobase_i', 'sag', 'vm_for_sag', 'input_resistance', 'tau',
              'fi_fit_slope', 'v_baseline'):
        val = analysis.get(k)
        if val is not None and np.isscalar(val) and np.isfinite(val):
            out[k] = float(val)

    spikes_set    = analysis.get('spikes_set')      # list[DataFrame], one per sweep
    spiking_sw    = analysis.get('spiking_sweeps')  # DataFrame of spiking sweeps

    # ── Per-sweep features (hero + rheobase) ─────────────────────────────────
    _SWEEP_SCALAR_COLS = ('avg_rate', 'stim_amp', 'v_baseline', 'sag',
                          'adapt', 'latency', 'isi_cv',
                          'mean_isi', 'median_isi', 'first_isi')
    _SPIKE_WAVEFORM_COLS = ('threshold_v', 'upstroke', 'downstroke',
                            'upstroke_downstroke_ratio', 'width',
                            'fast_trough_v', 'peak_v', 'adp_v')

    for sweep_key in ('hero_sweep', 'rheobase_sweep'):
        sweep = analysis.get(sweep_key)  # pandas Series
        if sweep is None:
            continue
        prefix = 'hero_' if sweep_key == 'hero_sweep' else 'rheobase_'

        # Sweep-level ISI / rate scalars
        for col in _SWEEP_SCALAR_COLS:
            if col in sweep.index:
                val = sweep[col]
                if val is not None and np.isscalar(val) and pd.notna(val):
                    out[prefix + col] = float(val)

        # First-spike AP waveform from spikes_set
        sweep_idx = int(sweep.name)  # integer position in spikes_set list
        if spikes_set is not None and sweep_idx < len(spikes_set):
            spk_df = spikes_set[sweep_idx]
            if len(spk_df) > 0:
                s0 = spk_df.iloc[0]
                for col in _SPIKE_WAVEFORM_COLS:
                    if col in spk_df.columns:
                        val = s0[col]
                        if pd.notna(val):
                            out[prefix + col] = float(val)
                # Derived: peak_deltav and fast_trough_deltav
                if 'peak_v' in spk_df.columns and 'threshold_v' in spk_df.columns:
                    pv, tv = s0['peak_v'], s0['threshold_v']
                    if pd.notna(pv) and pd.notna(tv):
                        out[prefix + 'peak_deltav'] = float(pv - tv)
                if 'fast_trough_v' in spk_df.columns and 'threshold_v' in spk_df.columns:
                    fv, tv = s0['fast_trough_v'], s0['threshold_v']
                    if pd.notna(fv) and pd.notna(tv):
                        out[prefix + 'fast_trough_deltav'] = float(fv - tv)

    # ── Cross-sweep adaptation features ──────────────────────────────────────
    if spiking_sw is not None and len(spiking_sw) > 0:
        # Mean adapt and ISI CV across all spiking sweeps
        for col, key in (('adapt', 'adapt_mean'), ('isi_cv', 'isi_cv_mean')):
            if col in spiking_sw.columns:
                vals = spiking_sw[col].dropna()
                if len(vals) > 0:
                    out[key] = float(vals.mean())

        # Adaptation ratios: first-spike waveform of last vs first spiking sweep
        # (sweeps sorted by stim_amp ascending)
        if spikes_set is not None and len(spiking_sw) >= 2:
            ss_sorted = spiking_sw.sort_values('stim_amp')
            first_idx = int(ss_sorted.index[0])
            last_idx  = int(ss_sorted.index[-1])

            def _first_spike_vals(sweep_idx, cols):
                if sweep_idx >= len(spikes_set):
                    return {}
                df = spikes_set[sweep_idx]
                if len(df) == 0:
                    return {}
                row = df.iloc[0]
                return {c: float(row[c]) for c in cols
                        if c in df.columns and pd.notna(row[c])}

            _RATIO_COLS = ('upstroke', 'downstroke', 'upstroke_downstroke_ratio',
                           'width', 'fast_trough_v', 'peak_v', 'threshold_v')
            first_vals = _first_spike_vals(first_idx, _RATIO_COLS)
            last_vals  = _first_spike_vals(last_idx,  _RATIO_COLS)

            for col in _RATIO_COLS:
                if col in first_vals and col in last_vals:
                    denom = first_vals[col]
                    if denom != 0 and np.isfinite(denom):
                        out[col + '_adapt_ratio'] = float(last_vals[col] / denom)

            # ISI adapt ratio: adapt of last vs first spiking sweep
            if 'adapt' in spiking_sw.columns:
                a_first = spiking_sw.loc[ss_sorted.index[0], 'adapt']
                a_last  = spiking_sw.loc[ss_sorted.index[-1], 'adapt']
                if pd.notna(a_first) and pd.notna(a_last) and a_first != 0:
                    out['isi_adapt_ratio'] = float(a_last / a_first)

    return out
from .nwb_recordings import get_pulse_times, get_intrinsic_recording_dict
from neuroanalysis.miesnwb import MiesNwb
from itertools import chain

import logging
logger = logging.getLogger(__name__)

def get_chirp_features(recordings, cell_id=''):
    errors = []
    if len(recordings) == 0:
        errors.append('No chirp sweeps for cell %s' % cell_id)
        return {}, errors
            
    sweep_list = []
    for rec in recordings:
        try:
            sweep = MPSweep(rec)
            sweep_list.append(sweep)
        except ValueError:
            continue
    
    if len(sweep_list) == 0:
        errors.append('No chirp sweeps passed qc for cell %s' % cell_id)
        return {}, errors

    # oodDAQ chirp sweeps may differ in length; truncate to the earliest end time
    # so that np.vstack in chirp_amp_phase gets equal-length arrays.
    min_end = min(s.t[-1] for s in sweep_list)
    if len(set(s.t[-1] for s in sweep_list)) > 1:
        truncated = []
        for s in sweep_list:
            end_idx = int(np.searchsorted(s.t, min_end, side='right'))
            if end_idx > 0:
                truncated.append(Sweep(s.t[:end_idx], s.v[:end_idx], s.i[:end_idx],
                                       s.clamp_mode, s.sampling_rate, sweep_number=s.sweep_number))
        if truncated:
            sweep_list = truncated

    sweep_set = SweepSet(sweep_list)
    # use actual sweep duration; fallback cap of 20.6 s may exceed real sweep length
    sweep_end = float(sweep_list[0].t[-1])
    try:
        chirp_features = extract_chirp_fft(sweep_set, min_freq=1, max_freq=min(15, sweep_end - 0.1),
                                           end=min(20.6, sweep_end - 0.05))
    except (FeatureError, Exception) as exc:
        logger.warning(f'Error processing chirps for cell {cell_id}: {str(exc)}')
        errors.append('Error processing chirps for cell %s: %s' % (cell_id, str(exc)))
        chirp_features = {}
    
    return chirp_features, errors

def get_long_square_features(recordings, cell_id=''):
    errors = []
    if len(recordings) == 0:
        errors.append('No long pulse sweeps for cell %s' % cell_id)
        return {}, errors

    min_pulse_dur = np.inf
    sweep_list = []
    for rec in recordings:
        try:
            pulse_times = get_pulse_times(rec)
            if pulse_times is None:
                raise ValueError("Pulse times not found for sweep.")
            # pulses may have different durations as well, so we just use the smallest duration
            start, end = pulse_times
            min_pulse_dur = min(min_pulse_dur, end-start)
            sweep = MPSweep(rec, -start)
            sweep_list.append(sweep)
        except ValueError:
            # report these errors?
            continue
    
    if len(sweep_list) == 0:
        errors.append('No long square sweeps passed qc for cell %s' % cell_id)
        return {}, errors

    sweep_set = SweepSet(sweep_list)
    spx, spfx = extractors_for_sweeps(sweep_set, start=0, end=min_pulse_dur)
    lsa = LongSquareAnalysis(spx, spfx, subthresh_min_amp=-200,
                                require_subthreshold=False, require_suprathreshold=False
                                )

    try:
        analysis = lsa.analyze(sweep_set)
    except FeatureError as exc:
        err = f'Error running long square analysis for cell {cell_id}: {str(exc)}'
        logger.warning(err)
        errors.append(err)
        return {}, errors

    # Work directly from the raw analysis dict (pandas objects intact).
    # as_dict() is not needed and discards the spikes_set required for
    # waveform features.
    output = get_complete_long_square_features(analysis)

    return output, errors

def features_from_nwb(filename, channels=None):
    nwb = MiesNwb(filename)
    channel_key = {}
    # need to convert from AD channel to device id
    for sweep in nwb.contents:
        for props in sweep._channel_keys.values():
            channel_key[int(props['AD'])] = int(props['ElectrodeName'])
    if channels is None:
        channels = channel_key.keys()

    records = list()
    for ch in channels:
        if ch not in channel_key:
            logger.warning(f'Channel {ch} not found in NWB file {filename}, skipping')
            continue
        dev = channel_key[ch]
        recording_dict = get_intrinsic_recording_dict(nwb, dev)
        
        results = dict(filename=filename, channel=ch, device=dev)
        if 'LP' in recording_dict:
            lp_results, error = get_long_square_features(recording_dict['LP'])
            results.update(lp_results)
        if 'Chirp' in recording_dict:
            chirp_results, error = get_chirp_features(recording_dict['Chirp'])
            results.update({feature+"_chirp": chirp_results.get(feature) for feature in chirp_results.keys()})
        records.append(results)
    return records

def process_file_list(files, channels_list=None):
    if channels_list:
        records = chain(*(features_from_nwb(file, channels) 
                          for file, channels in zip(files, channels_list)))
    else:
        records = chain(*(features_from_nwb(file) for file in files))
    return records

class MPSweep(Sweep):
    """Adapter for neuroanalysis.Recording => ipfx.Sweep
    """
    def __init__(self, rec, t0=0):
        # pulses may have different start times, so we shift time values to make all pulses start at t=0
        pri = rec['primary'].copy(t0=t0)
        cmd = rec['command'].copy()
        t = pri.time_values
        v = pri.data * 1e3  # convert to mV
        holding = [i for i in rec.stimulus.items if i.description=='holding current']
        if len(holding) == 0:
            raise ValueError("Sweep missing holding current data.")
        holding = holding[0].amplitude
        i = (cmd.data - holding) * 1e12   # convert to pA with holding current removed
        srate = pri.sample_rate
        sweep_num = rec.parent.key
        # modes 'ic' and 'vc' should be expanded
        clamp_mode = "CurrentClamp" if rec.clamp_mode=="ic" else "VoltageClamp"

        valid_data = (v != 0) & ~np.isnan(v)
        if not np.any(valid_data):
            raise ValueError("All-zero / NaN sweep.")

        # Use a contiguous slice from first→last valid sample so the time axis
        # stays uniformly sampled.  Filtering by index (original approach) creates
        # gaps that break ipfx's assumption of uniform sampling.
        valid_idx = np.where(valid_data)[0]
        first_v, last_v = valid_idx[0], valid_idx[-1]
        t_trim = t[first_v:last_v + 1]
        v_trim = v[first_v:last_v + 1]
        i_trim = i[first_v:last_v + 1]

        # not sure where exactly to put this cutoff - maybe need to refine
        if t_trim[-1] < 0.5:
            raise ValueError("Incomplete / nan sweep.")

        Sweep.__init__(self, t_trim, v_trim, i_trim, clamp_mode, srate, sweep_number=sweep_num)