#include "fft_features.h"
#include <math.h>

/* Log-spaced band edges in Hz, 9 edges -> 8 bands. Placeholder — see
 * header comment. Must match training/data_prep/preprocess.py's
 * BAND_EDGES_HZ exactly. */
static const float BAND_EDGES_HZ[FFT_N_BANDS + 1] = {
    5.0f, 8.9f, 15.8f, 28.1f, 50.0f, 88.9f, 158.1f, 281.2f, 400.0f
};

void FFTFeatures_Init(FFTFeatures_Handle_t *h)
{
    int i;
    arm_rfft_fast_init_f32(&h->rfft, FFT_WINDOW_SIZE);

    /* Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1))) */
    for (i = 0; i < FFT_WINDOW_SIZE; i++) {
        h->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * PI * (float)i / (float)(FFT_WINDOW_SIZE - 1)));
    }
}

static void ComputeMagnitudeSpectrum(FFTFeatures_Handle_t *h)
{
    /* arm_rfft_fast_f32 packs real-FFT output as
     * [Re(0), Re(N/2), Re(1), Im(1), Re(2), Im(2), ...] — bin 0 (DC) and
     * bin N/2 (Nyquist) are real-only and packed into the first two
     * float32 slots. Bins 1..N/2-1 are proper complex pairs starting at
     * index 2, which arm_cmplx_mag_f32 handles directly. */
    arm_rfft_fast_f32(&h->rfft, h->fft_buf, h->fft_out, 0 /* forward transform */);

    h->mag_buf[0] = fabsf(h->fft_out[0]); /* DC bin */
    arm_cmplx_mag_f32(&h->fft_out[2], &h->mag_buf[1], (FFT_WINDOW_SIZE / 2) - 1);
}

static void BandEnergiesFromMagnitude(FFTFeatures_Handle_t *h, float sample_rate_hz, float *out_bands)
{
    float bin_hz = sample_rate_hz / (float)FFT_WINDOW_SIZE;
    int band;

    for (band = 0; band < FFT_N_BANDS; band++) {
        int bin_lo = (int)(BAND_EDGES_HZ[band] / bin_hz);
        int bin_hi = (int)(BAND_EDGES_HZ[band + 1] / bin_hz);
        float energy = 0.0f;
        int bin;

        if (bin_hi >= FFT_WINDOW_SIZE / 2) bin_hi = (FFT_WINDOW_SIZE / 2) - 1;
        for (bin = bin_lo; bin <= bin_hi; bin++) {
            energy += h->mag_buf[bin] * h->mag_buf[bin];
        }
        out_bands[band] = energy;
    }
}

void FFTFeatures_Extract_i16(FFTFeatures_Handle_t *h, const int16_t *window,
                              float sample_rate_hz, float *out_bands)
{
    int i;
    for (i = 0; i < FFT_WINDOW_SIZE; i++) {
        h->fft_buf[i] = (float32_t)window[i] * h->hann_window[i];
    }
    ComputeMagnitudeSpectrum(h);
    BandEnergiesFromMagnitude(h, sample_rate_hz, out_bands);
}

void FFTFeatures_Extract_f32(FFTFeatures_Handle_t *h, const float32_t *window,
                              float sample_rate_hz, float *out_bands)
{
    int i;
    for (i = 0; i < FFT_WINDOW_SIZE; i++) {
        h->fft_buf[i] = window[i] * h->hann_window[i];
    }
    ComputeMagnitudeSpectrum(h);
    BandEnergiesFromMagnitude(h, sample_rate_hz, out_bands);
}
