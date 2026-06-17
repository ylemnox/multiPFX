"""
Standalone intrinsic property extraction from multipatch NWB files.

Usage
-----
1. Place your NWB files (or subdirectories of NWB files) inside the `data/` folder.
2. Place your metadata Excel file inside `data/` and set EXCEL_PATH below.
3. Run:  python extract_intrinsic.py
4. Output → intrinsic_features.csv in this directory.

Dependencies (same conda/venv):
    pip install ipfx neuroanalysis h5py pandas openpyxl
    cd aisynphys && pip install -e .
"""

import re
import logging
from pathlib import Path
from collections import defaultdict

import h5py
import pandas as pd
from aisynphys.intrinsic_ephys import features_from_nwb
import neuroanalysis.miesnwb as _miesnwb_mod
from neuroanalysis.miesnwb import MiesNwb, MiesRecording, MiesTSeries, MiesStimulus
from neuroanalysis.data import PatchClampRecording
from neuroanalysis import stimuli


def _patch_miesnwb_for_nwb2():
    """Monkey-patch MiesNwb and MiesRecording to handle NWB 2.x MIES files."""
    _orig_contents = MiesNwb.contents.fget

    def _parse_electrode_map(comments):
        """Return (adc_to_electrode, adc_to_dac) dicts from MIES NWB 2.x comments string."""
        adc_to_hs = {}
        hs_to_electrode = {}
        hs_to_dac = {}
        for line in comments.split('\n'):
            m = re.match(r'HS#(\d+):ADC:\s*([\d.]+)', line)
            if m:
                adc_to_hs[int(float(m.group(2)))] = int(m.group(1))
            m = re.match(r'HS#(\d+):Electrode:\s*(\d+)', line)
            if m:
                hs_to_electrode[int(m.group(1))] = int(m.group(2))
            m = re.match(r'HS#(\d+):DAC:\s*([\d.]+)', line)
            if m:
                hs_to_dac[int(m.group(1))] = int(float(m.group(2)))
        adc_to_electrode = {adc: hs_to_electrode.get(hs, adc) for adc, hs in adc_to_hs.items()}
        adc_to_dac = {adc: hs_to_dac.get(hs, adc) for adc, hs in adc_to_hs.items()}
        return adc_to_electrode, adc_to_dac, adc_to_hs

    @property
    def _patched_contents(self):
        if self._sweeps is not None:
            return self._sweeps

        hdf = self.hdf
        raw = hdf._wrapped_obj if hasattr(hdf, '_wrapped_obj') else hdf

        # NWB 1.x path
        if 'acquisition/timeseries' in raw:
            return _orig_contents(self)

        # NWB 2.x path
        self._timeseries = {}
        acq = raw['acquisition']

        electrode_map = {}
        dac_map = {}
        hs_map = {}
        for entry_name, entry in acq.items():
            comments = entry.attrs.get('comments', '')
            if comments:
                electrode_map, dac_map, hs_map = _parse_electrode_map(comments)
                break

        # Store dac_map for use in da_chan()
        self._nwb2_dac_map = dac_map

        for ts_name, ts in raw['acquisition'].items():
            m = re.match(r'data_(\d+)_AD(\d+)$', ts_name)
            if not m:
                continue
            sweep = int(m.group(1))
            ad_chan = int(m.group(2))
            hs_num = hs_map.get(ad_chan, ad_chan)
            src = {
                'Sweep': str(sweep),
                'AD': str(ad_chan),
                'ElectrodeName': str(electrode_map.get(ad_chan, ad_chan)),
                'ElectrodeNumber': str(electrode_map.get(ad_chan, ad_chan)),
                'HeadstageNumber': str(hs_num),  # HS# for notebook column lookup
                'Device': 'ITC1600_Dev_0',
                'hdf_group_name': 'acquisition/' + ts_name,
            }
            self._timeseries.setdefault(sweep, {})[ad_chan] = src

        sweep_ids = sorted(self._timeseries.keys())
        self._sweeps = []
        for sweep_id in sweep_ids:
            try:
                srec = self.create_sync_recording(int(sweep_id))
                self._sweeps.append(srec)
            except Exception:
                pass  # skip sweeps whose recording cannot be initialised
        return self._sweeps

    MiesNwb.contents = _patched_contents

    # ── Patch MiesRecording.__init__ ──────────────────────────────────────────

    def _patched_MiesRecording_init(self, sweep, sweep_id, ad_chan):
        self._sweep = sweep
        self._nwb = sweep._nwb
        self._trace_id = (sweep_id, ad_chan)
        self._inserted_test_pulse = None
        self._nearest_test_pulse = None
        self._hdf_group_name = sweep._channel_keys[ad_chan]['hdf_group_name']
        self._hdf_group = None
        self._da_chan = None

        notebook_id = None
        try:
            elec_bytes = self.hdf_group['electrode_name'][()][0]
            elec = elec_bytes.decode() if hasattr(elec_bytes, 'decode') else elec_bytes
            headstage_id = int(elec.split('_')[1])
            notebook_id = headstage_id  # NWB 1.x: electrode# == notebook column
        except (KeyError, OSError, IndexError):
            # NWB 2.x: electrode_name not present; use ElectrodeName as device_id
            headstage_id = int(sweep._channel_keys[ad_chan]['ElectrodeName'])
            # Use HeadstageNumber (HS#) for notebook column — may differ from Electrode#
            notebook_id = int(sweep._channel_keys[ad_chan].get('HeadstageNumber', headstage_id))

        PatchClampRecording.__init__(self, device_type='MultiClamp 700', device_id=headstage_id,
                                     sync_recording=sweep)

        notebook_entry = self._nwb.notebook().get(int(self._trace_id[0]))
        if notebook_entry is None:
            raise ValueError(f"No notebook entry for sweep {self._trace_id[0]}")
        # Defend against off-by-one electrode numbering: try notebook_id, then notebook_id-1
        if notebook_id < len(notebook_entry):
            nb = notebook_entry[notebook_id]
        elif notebook_id - 1 >= 0 and notebook_id - 1 < len(notebook_entry):
            nb = notebook_entry[notebook_id - 1]
        else:
            raise ValueError(f"Notebook index {notebook_id} out of range (len={len(notebook_entry)})")
        self.meta['holding_potential'] = (
            None if nb['V-Clamp Holding Level'] is None
            else nb['V-Clamp Holding Level'] * 1e-3
        )
        self.meta['holding_current'] = (
            None if nb['I-Clamp Holding Level'] is None
            else nb['I-Clamp Holding Level'] * 1e-12
        )
        self._meta['notebook'] = nb
        if nb['Clamp Mode'] == 0:
            self._meta['clamp_mode'] = 'vc'
            primary_units = 'A'
            command_units = 'V'
        else:
            self._meta['clamp_mode'] = 'ic'
            self._meta['bridge_balance'] = (
                0.0 if nb['Bridge Bal Enable'] == 0.0 or nb['Bridge Bal Value'] is None
                else nb['Bridge Bal Value'] * 1e6
            )
            primary_units = 'V'
            command_units = 'A'
        self._meta['lpf_cutoff'] = nb['LPF Cutoff']
        offset = nb['Pipette Offset']
        self._meta['pipette_offset'] = None if offset is None else offset * 1e-3
        self.meta['start_time'] = MiesNwb.igorpro_date(nb['TimeStamp'])

        self._channels['primary'] = MiesTSeries(self, 'primary', units=primary_units)
        self._channels['command'] = MiesTSeries(self, 'command', units=command_units)

    MiesRecording.__init__ = _patched_MiesRecording_init

    # ── Patch MiesRecording.da_chan() ─────────────────────────────────────────

    def _patched_da_chan(self):
        if self._da_chan is not None:
            return self._da_chan

        # NWB 1.x: scan stimulus/presentation for matching electrode_name
        try:
            hdf = self._nwb.hdf['stimulus/presentation']
            raw_hdf = hdf._wrapped_obj if hasattr(hdf, '_wrapped_obj') else hdf
            prefix = 'data_%05d_' % self._trace_id[0]
            for s in raw_hdf.keys():
                if not s.startswith(prefix):
                    continue
                grp = raw_hdf[s]
                if 'electrode_name' not in grp:
                    continue
                elec = grp['electrode_name'][()][0]
                if hasattr(elec, 'decode'):
                    elec = elec.decode()
                if elec == 'electrode_%d' % self.device_id:
                    self._da_chan = int(s.split('_')[-1][2:])
                    return self._da_chan
        except Exception:
            pass

        # NWB 2.x: use dac_map built from comments
        dac_map = getattr(self._nwb, '_nwb2_dac_map', {})
        ad_chan = self._trace_id[1]
        if ad_chan in dac_map:
            self._da_chan = dac_map[ad_chan]
            return self._da_chan

        # Last resort: DA == AD (common MIES default)
        self._da_chan = ad_chan
        return self._da_chan

    MiesRecording.da_chan = _patched_da_chan

    # ── Patch MiesStimulus.__init__ ───────────────────────────────────────────
    # NWB 1.x stores stimulus_description as an HDF5 dataset inside the group.
    # NWB 2.x stores it as an HDF5 attribute on the group.

    def _patched_MiesStimulus_init(self, recording):
        self._recording = recording
        try:
            raw = recording.hdf_group['stimulus_description'][()][0]
            stim_name = raw.decode() if hasattr(raw, 'decode') else raw
        except (KeyError, OSError):
            stim_name = recording.hdf_group.attrs.get('stimulus_description', '')
        stimuli.Stimulus.__init__(self, description=stim_name)
        self._items = None

    MiesStimulus.__init__ = _patched_MiesStimulus_init

    # ── Patch MiesRecording.inserted_test_pulse ────────────────────────────────
    # In oodDAQ mode the command waveform for each channel contains multiple
    # square pulses, so PatchClampTestPulse raises ValueError even when
    # slice indices are provided. Return None instead so _parse_wavenote skips it.

    from neuroanalysis.test_pulse import PatchClampTestPulse

    @property
    def _patched_inserted_test_pulse(self):
        if self._inserted_test_pulse is None:
            if not self.has_inserted_test_pulse:
                return None
            pulse_dur = self.meta['notebook']['TP Pulse Duration'] / 1000.
            total_dur = pulse_dur / (1.0 - 2. * self.meta['notebook']['TP Baseline Fraction'])
            start = 0
            stop = start + int(total_dur / self['primary'].dt)
            try:
                tp = PatchClampTestPulse(self, indices=(start, stop))
            except ValueError:
                return None
            if self.clamp_mode == 'vc':
                amp = self.meta['notebook']['TP Amplitude VC'] * 1e-3
            else:
                amp = self.meta['notebook']['TP Amplitude IC'] * 1e-12
            self._inserted_test_pulse = tp
        return self._inserted_test_pulse

    MiesRecording.inserted_test_pulse = _patched_inserted_test_pulse

    # ── Patch MiesStimulus._parse_wavenote to guard against None test pulse ───

    _orig_parse_wavenote = MiesStimulus._parse_wavenote

    def _patched_parse_wavenote(self):
        rec = self._recording
        # guard the inserted_test_pulse access against None (oodDAQ multi-pulse)
        _orig_has_itp = rec.has_inserted_test_pulse
        if _orig_has_itp and rec.inserted_test_pulse is None:
            # temporarily fake has_inserted_test_pulse = False so the original
            # code's guard passes cleanly
            rec._meta['notebook'] = dict(rec._meta['notebook'])
            rec._meta['notebook']['TP Insert Checkbox'] = 0.0
            try:
                return _orig_parse_wavenote(self)
            finally:
                rec._meta['notebook']['TP Insert Checkbox'] = 1.0
        return _orig_parse_wavenote(self)

    MiesStimulus._parse_wavenote = _patched_parse_wavenote


_patch_miesnwb_for_nwb2()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
EXCEL_PATH  = _HERE / 'data' / 'meta_info.xlsx'   # ← rename your Excel file to this
NWB_ROOT    = _HERE / 'data'                        # ← NWB files go here
OUTPUT_CSV  = _HERE / 'intrinsic_features.csv'
# ─────────────────────────────────────────────────────────────────────────────


def parse_excel(excel_path):
    """Return {timestamp: {'channels': set, 'sheet': str}} from all sheets."""
    xl = pd.ExcelFile(excel_path)
    ts_map = defaultdict(lambda: {'channels': set(), 'sheet': ''})
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=0)
        col0 = df.columns[0]
        for _, row in df.iterrows():
            name = str(row[col0]).strip("' ")
            m = re.match(r'(\d{4}_\d{2}_\d{2}_\d{6})', name)
            if not m:
                continue
            ts = m.group(1)
            channels = [int(x) for x in re.findall(r'AD(\d+)', name)]
            ts_map[ts]['channels'].update(channels)
            ts_map[ts]['sheet'] = sheet
    return ts_map


def build_nwb_index(nwb_root):
    """Return {timestamp: Path} for every .nwb file under nwb_root.
    Prefers nwb_v1 files over nwb_v2 when both exist for the same timestamp.
    """
    v1, v2 = {}, {}
    for f in nwb_root.rglob('*.nwb'):
        ts = f.name.split('_ITC')[0].split('-')[0]
        if not re.match(r'\d{4}_\d{2}_\d{2}_\d{6}$', ts):
            continue
        if 'nwb_v1' in f.parts:
            v1[ts] = f
        else:
            v2[ts] = f
    # merge, v1 wins
    return {**v2, **v1}


def main():
    log.info('Parsing Excel metadata ...')
    ts_map = parse_excel(EXCEL_PATH)

    log.info('Indexing NWB files ...')
    nwb_index = build_nwb_index(NWB_ROOT)

    matched = []
    unmatched = []
    for ts, info in ts_map.items():
        if ts in nwb_index:
            matched.append((ts, nwb_index[ts], sorted(info['channels']), info['sheet']))
        else:
            unmatched.append(ts)

    log.info(f'Matched {len(matched)} files, {len(unmatched)} not found on drive')
    if unmatched:
        log.warning(f'Missing NWB files for: {unmatched}')

    all_records = []
    for i, (ts, nwb_path, channels, sheet) in enumerate(matched):
        log.info(f'[{i+1}/{len(matched)}] {nwb_path.name}  channels={channels}  ({sheet})')
        try:
            records = features_from_nwb(nwb_path, channels if channels else None)
            for rec in records:
                rec['sheet'] = sheet
                rec['timestamp'] = ts
            all_records.extend(records)
            log.info(f'  -> {len(records)} cell(s) extracted')
        except Exception as e:
            import traceback
            log.error(f'  -> FAILED: {e}\n' + traceback.format_exc())

    if not all_records:
        log.error('No records extracted — check errors above.')
        return

    df = pd.DataFrame.from_records(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f'Saved {len(df)} rows to {OUTPUT_CSV}')
    log.info(f'Features: {list(df.columns)}')


if __name__ == '__main__':
    main()
