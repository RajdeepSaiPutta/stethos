#!/usr/bin/env python3
"""
Realistic breath sound synthesis based on respiratory acoustics literature.
Models: vesicular, bronchial, crackles (fine/coarse), wheezes (monophonic/polyphonic),
rhonchi, stridor, pleural rub. Includes realistic breathing cycle envelope,
background noise, and microphone characteristics.
"""

import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
from pathlib import Path
import argparse


SR = 16000
DURATION = 8.0  # Longer for full breathing cycles


def butter_bandpass(low: float, high: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, [low/nyq, high/nyq], btype='band')


def apply_filter(x: np.ndarray, low: float, high: float, fs: int) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs)
    return filtfilt(b, a, x)


def breathing_envelope(t: np.ndarray, rate_bpm: float = 12, ie_ratio: float = 1/2) -> np.ndarray:
    """Realistic breathing cycle envelope with inspiratory/expiratory phases."""
    cycle = 60.0 / rate_bpm
    t_mod = t % cycle
    t_insp = cycle * ie_ratio / (1 + ie_ratio)
    
    env = np.zeros_like(t)
    insp_mask = t_mod < t_insp
    exp_mask = ~insp_mask
    
    # Inspiratory: smooth rise, slightly longer
    env[insp_mask] = np.sin(np.pi * t_mod[insp_mask] / t_insp) ** 1.5
    # Expiratory: faster decay
    env[exp_mask] = np.exp(-3 * (t_mod[exp_mask] - t_insp) / (cycle - t_insp))
    
    # Add slight inter-breath pause
    pause_start = cycle * 0.95
    pause_mask = t_mod > pause_start
    env[pause_mask] *= np.exp(-20 * (t_mod[pause_mask] - pause_start) / (cycle - pause_start))
    
    return 0.3 + 0.7 * env  # Never fully silent (background body sounds)


def vesicular_sound(t: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Normal vesicular breath sounds: soft, low-pitched, inspiratory > expiratory."""
    # Broadband noise filtered to 50-400 Hz
    noise = np.random.randn(len(t))
    noise = apply_filter(noise, 50, 400, SR)
    
    # Add low-frequency "rumble" from turbulent flow
    rumble = 0.3 * np.sin(2 * np.pi * 80 * t) * env
    rumble += 0.2 * np.sin(2 * np.pi * 160 * t) * env
    
    # Inspiratory louder than expiratory
    insp_gain = np.where(t % (60/12) < (60/12) * (1/3), 1.2, 0.6)
    
    return (noise * 0.4 + rumble) * env * insp_gain


def bronchial_sound(t: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Bronchial breath sounds: louder, higher-pitched, equal I/E, hollow quality."""
    noise = np.random.randn(len(t))
    noise = apply_filter(noise, 100, 800, SR)
    
    # Hollow resonance
    hollow = 0.4 * np.sin(2 * np.pi * 200 * t) * env
    hollow += 0.3 * np.sin(2 * np.pi * 400 * t) * env
    
    return (noise * 0.6 + hollow) * env


def crackles(t: np.ndarray, env: np.ndarray, type_: str = 'fine') -> np.ndarray:
    """Discontinuous adventitious sounds (crackles/rales).
    Fine: high-freq (400-800Hz), short (2-10ms), late inspiratory
    Coarse: lower-freq (200-400Hz), longer (10-25ms), early inspiratory
    """
    audio = np.zeros_like(t)
    cycle = 60/12
    
    if type_ == 'fine':
        freq_range = (400, 800)
        dur_range = (0.002, 0.01)
        rate_per_cycle = 8
        timing = 'late_insp'  # Last 30% of inspiration
    else:  # coarse
        freq_range = (200, 400)
        dur_range = (0.01, 0.025)
        rate_per_cycle = 3
        timing = 'early_insp'  # First 50% of inspiration
    
    for cycle_idx in range(int(len(t) / SR / cycle) + 1):
        cycle_start = int(cycle_idx * cycle * SR)
        cycle_end = min(len(t), int((cycle_idx + 1) * cycle * SR))
        if cycle_start >= len(t):
            break
            
        t_insp = cycle * (1/3)
        insp_end = int(t_insp * SR)
        
        if timing == 'late_insp':
            win_start = cycle_start + int(0.7 * insp_end)
            win_end = cycle_start + insp_end
        else:
            win_start = cycle_start
            win_end = cycle_start + int(0.5 * insp_end)
        
        win_start = max(cycle_start, win_start)
        win_end = min(cycle_end, win_end)
        if win_end <= win_start:
            continue
            
        n_crackles = np.random.poisson(rate_per_cycle)
        for _ in range(n_crackles):
            pos = np.random.randint(win_start, min(win_end - int(SR*0.03), len(t) - int(SR*0.03)))
            dur = int(np.random.uniform(*dur_range) * SR)
            freq = np.random.uniform(*freq_range)
            
            # Damped sinusoid
            crackle_t = np.arange(dur) / SR
            decay = np.exp(-crackle_t / (0.002 if type_ == 'fine' else 0.005))
            crackle = decay * np.sin(2 * np.pi * freq * crackle_t)
            
            # Window to avoid clicks
            window = np.hanning(dur)
            crackle *= window
            
            audio[pos:pos+dur] += crackle * env[pos] * 0.8
    
    return audio


def wheezes(t: np.ndarray, env: np.ndarray, type_: str = 'monophonic') -> np.ndarray:
    """Continuous adventitious sounds (wheezes/rhonchi).
    Monophonic: single frequency, sinusoidal, >100ms
    Polyphonic: multiple frequencies, musical quality
    Rhonchi: low-pitched (<200Hz), snoring quality
    """
    audio = np.zeros_like(t)
    cycle = 60/12
    
    if type_ == 'monophonic':
        base_freq = np.random.uniform(200, 500)
        fm_depth = 20
        fm_rate = 0.5
        duration = DURATION
    elif type_ == 'polyphonic':
        base_freq = np.random.uniform(150, 400)
        fm_depth = 50
        fm_rate = 1.0
        duration = DURATION
    else:  # rhonchi
        base_freq = np.random.uniform(80, 180)
        fm_depth = 10
        fm_rate = 0.3
        duration = DURATION
    
    # Frequency modulation for musical quality
    fm = fm_depth * np.sin(2 * np.pi * fm_rate * t)
    inst_freq = base_freq + fm
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    
    # Amplitude modulation by breathing
    carrier = np.sin(phase)
    audio = carrier * env * 0.5
    
    # Add harmonics for polyphonic
    if type_ == 'polyphonic':
        for h in [2, 3]:
            audio += 0.3/h * np.sin(h * phase) * env
    
    return audio


def pleural_rub(t: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Pleural friction rub: grating, creaking, biphasic (insp+exp)."""
    audio = np.zeros_like(t)
    # Series of short bursts in both phases
    cycle = 60/12
    for cycle_idx in range(int(len(t) / SR / cycle) + 1):
        cycle_start = int(cycle_idx * cycle * SR)
        cycle_end = min(len(t), int((cycle_idx + 1) * cycle * SR))
        if cycle_start >= len(t):
            break
        
        for phase in ['insp', 'exp']:
            if phase == 'insp':
                ph_start = cycle_start
                ph_end = cycle_start + int(cycle * (1/3) * SR)
            else:
                ph_start = cycle_start + int(cycle * (1/3) * SR)
                ph_end = cycle_end
            
            n_rubs = np.random.poisson(4)
            for _ in range(n_rubs):
                pos = np.random.randint(ph_start, min(ph_end - int(0.02*SR), len(t) - int(0.02*SR)))
                dur = int(np.random.uniform(0.02, 0.08) * SR)
                
                # Creaking sound: bandpass noise + low freq modulation
                rub = np.random.randn(dur)
                rub = apply_filter(rub, 50, 300, SR)
                # Slow amplitude modulation
                mod = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * np.arange(dur) / SR)
                rub *= mod * np.hanning(dur)
                
                audio[pos:pos+dur] += rub * env[pos] * 0.6
    
    return audio


def stridor(t: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Stridor: high-pitched, inspiratory, indicates upper airway obstruction."""
    audio = np.zeros_like(t)
    cycle = 60/12
    base_freq = np.random.uniform(400, 800)
    
    for cycle_idx in range(int(len(t) / SR / cycle) + 1):
        cycle_start = int(cycle_idx * cycle * SR)
        insp_end = cycle_start + int(cycle * (1/3) * SR)
        if cycle_start >= len(t):
            break
        
        # Only during inspiration
        t_seg = np.arange(insp_end - cycle_start) / SR
        fm = 30 * np.sin(2 * np.pi * 2 * t_seg)
        phase = 2 * np.pi * np.cumsum(base_freq + fm) / SR
        carrier = np.sin(phase)
        
        # Harsh quality - add harmonics
        audio[cycle_start:insp_end] += carrier * env[cycle_start:insp_end] * 0.7
        audio[cycle_start:insp_end] += 0.5 * np.sin(2*phase) * env[cycle_start:insp_end] * 0.5
    
    return audio


def add_background_noise(audio: np.ndarray, snr_db: float = 30) -> np.ndarray:
    """Add realistic ambient/body noise."""
    noise = np.random.randn(len(audio))
    noise = apply_filter(noise, 20, 200, SR)  # Low-frequency body sounds
    noise_rms = np.sqrt(np.mean(noise**2))
    signal_rms = np.sqrt(np.mean(audio**2))
    if signal_rms > 0:
        target_noise_rms = signal_rms / (10**(snr_db/20))
        noise *= target_noise_rms / (noise_rms + 1e-10)
    return audio + noise


def microphone_response(audio: np.ndarray) -> np.ndarray:
    """Simulate stethoscope/microphone frequency response."""
    # Stethoscope bell: emphasis 50-200Hz, rolloff above 400Hz
    # Diaphragm: flatter to 1000Hz
    # Simulate diaphragm response
    b, a = butter(2, [50/8000, 1000/8000], btype='band')
    return filtfilt(b, a, audio)


def generate_case(out_path: Path, case_type: str, seed: int = None):
    """Generate a realistic breath sound case."""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(int(SR * DURATION)) / SR
    env = breathing_envelope(t, rate_bpm=np.random.uniform(10, 16), ie_ratio=np.random.uniform(0.4, 0.55))
    
    audio = np.zeros_like(t)
    
    if case_type == 'normal':
        audio += vesicular_sound(t, env)
        
    elif case_type == 'crackle_fine':
        audio += vesicular_sound(t, env) * 0.7
        audio += crackles(t, env, 'fine')
        
    elif case_type == 'crackle_coarse':
        audio += vesicular_sound(t, env) * 0.7
        audio += crackles(t, env, 'coarse')
        
    elif case_type == 'wheeze_monophonic':
        audio += vesicular_sound(t, env) * 0.5
        audio += wheezes(t, env, 'monophonic')
        
    elif case_type == 'wheeze_polyphonic':
        audio += vesicular_sound(t, env) * 0.5
        audio += wheezes(t, env, 'polyphonic')
        
    elif case_type == 'rhonchi':
        audio += vesicular_sound(t, env) * 0.5
        audio += wheezes(t, env, 'rhonchi')
        
    elif case_type == 'bronchial':
        audio += bronchial_sound(t, env)
        
    elif case_type == 'pleural_rub':
        audio += vesicular_sound(t, env) * 0.6
        audio += pleural_rub(t, env)
        
    elif case_type == 'stridor':
        audio += vesicular_sound(t, env) * 0.4
        audio += stridor(t, env)
        
    elif case_type == 'copd':  # Mixed obstructive pattern
        audio += vesicular_sound(t, env) * 0.4
        audio += wheezes(t, env, 'polyphonic') * 0.8
        audio += crackles(t, env, 'coarse') * 0.6
        audio += wheezes(t, env, "rhonchi") * 0.5
        
    elif case_type == 'pneumonia':  # Consolidation -> bronchial + crackles
        audio += bronchial_sound(t, env) * 0.7
        audio += crackles(t, env, 'fine') * 0.9
        audio += crackles(t, env, 'coarse') * 0.5
        
    elif case_type == 'interstitial':  # Fine crackles dominant
        audio += vesicular_sound(t, env) * 0.5
        audio += crackles(t, env, 'fine') * 1.2
        
    elif case_type == 'heart_failure':  # Coarse crackles + wheeze
        audio += vesicular_sound(t, env) * 0.5
        audio += crackles(t, env, 'coarse') * 1.0
        audio += wheezes(t, env, 'monophonic') * 0.5
        
    else:
        raise ValueError(f"Unknown case type: {case_type}")
    
    # Post-processing
    audio = add_background_noise(audio, snr_db=np.random.uniform(25, 35))
    audio = microphone_response(audio)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.85
    audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(str(out_path), SR, audio)
    return {"path": str(out_path), "type": case_type, "duration": DURATION, "sr": SR}


CASE_TYPES = [
    'normal', 'crackle_fine', 'crackle_coarse', 'wheeze_monophonic',
    'wheeze_polyphonic', 'rhonchi', 'bronchial', 'pleural_rub',
    'stridor', 'copd', 'pneumonia', 'interstitial', 'heart_failure'
]

# Clinical labels mapping
LABEL_MAP = {
    'normal': 'normal',
    'crackle_fine': 'crackle',
    'crackle_coarse': 'crackle',
    'wheeze_monophonic': 'wheeze',
    'wheeze_polyphonic': 'wheeze',
    'rhonchi': 'wheeze',  # Low-pitched wheeze
    'bronchial': 'normal',  # Not adventitious, but abnormal pattern
    'pleural_rub': 'crackle',  # Closest category
    'stridor': 'wheeze',
    'copd': 'both',
    'pneumonia': 'crackle',
    'interstitial': 'crackle',
    'heart_failure': 'both',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="test-data/breath_realistic")
    parser.add_argument("--n-per-type", type=int, default=3)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels = {}
    for case_type in CASE_TYPES:
        for i in range(args.n_per_type):
            fname = f"{case_type}_{i+1:02d}.wav"
            fpath = out_dir / fname
            meta = generate_case(fpath, case_type, seed=hash(fname) % 2**32)
            labels[fname] = {
                "clinical_label": LABEL_MAP[case_type],
                "synthesis_type": case_type,
                "duration": DURATION,
                "sr": SR
            }
    
    with open(out_dir / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Generated {len(labels)} realistic breath sounds in {out_dir}")
    for ct in CASE_TYPES:
        print(f"  {ct}: {args.n_per_type} samples")