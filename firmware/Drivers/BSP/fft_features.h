#ifndef FFT_FEATURES_H
#define FFT_FEATURES_H

#include "arm_math.h"
#include <stdint.h>

/*
 * On-device FFT + band-energy feature extraction (pipeline step 3), using
 * CMSIS-DSP. Requires CMSIS-DSP vendored into firmware/Drivers/CMSIS-DSP/
 * and arm_math.h on the include path — not done yet, see that folder's
 * README. This file will not compile until that's in place.
 *
 * Band edges below are placeholders — 8 log-spaced bands from 5Hz to
 * 400Hz (Nyquist at SAMPLE_RATE_HZ=800 from main.h). Nothing about these
 * numbers is derived from real fault data, because none exists yet.
 * Revisit once data/raw/ has real logged runs: plot the FFT of each fault
 * class and pick bands that actually separate them, don't just trust these.
 *
 * CRITICAL: FFT_WINDOW_SIZE and BAND_EDGES_HZ (in the .c file) must exactly
 * match training/data_prep/preprocess.py's window size and band
 * definitions, or offline accuracy won't transfer to the on-device model.
 * There is no shared source of truth between the two right now — if you
 * change one, change the other by hand.
 *
 * Also verify arm_rfft_fast_f32's exact signature against whatever
 * CMSIS-DSP version actually gets vendored — it has changed slightly
 * across versions (return type in particular). Written here against the
 * commonly-documented void-returning signature; double check before
 * assuming this compiles as-is.
 */

#define FFT_WINDOW_SIZE 256   /* must equal WINDOW_SIZE_SAMPLES in main.h */
#define FFT_N_BANDS     8

typedef struct {
    arm_rfft_fast_instance_f32 rfft;
    float32_t hann_window[FFT_WINDOW_SIZE];
    float32_t fft_buf[FFT_WINDOW_SIZE];      /* windowed input, reused as FFT scratch */
    float32_t fft_out[FFT_WINDOW_SIZE];      /* FFT output, CMSIS packed format */
    float32_t mag_buf[FFT_WINDOW_SIZE / 2];  /* magnitude spectrum, bins 0..N/2-1 */
} FFTFeatures_Handle_t;

/* One-time init: builds the Hann window table and the CMSIS-DSP FFT
 * instance. Call once at startup, not per-window — arm_rfft_fast_init_f32
 * does nontrivial setup work. */
void FFTFeatures_Init(FFTFeatures_Handle_t *h);

/* Runs FFT + band-energy extraction on one window of int16 samples.
 * out_bands must point to FFT_N_BANDS floats. sample_rate_hz is passed
 * explicitly (not hardcoded) so this can be reused for channels sampled
 * at different rates later. */
void FFTFeatures_Extract_i16(FFTFeatures_Handle_t *h, const int16_t *window,
                              float sample_rate_hz, float *out_bands);

/* Same as above but for float32 input (used for the current channel,
 * which is already float from ACS712_ReadCurrent_mA) */
void FFTFeatures_Extract_f32(FFTFeatures_Handle_t *h, const float32_t *window,
                              float sample_rate_hz, float *out_bands);

#endif /* FFT_FEATURES_H */
