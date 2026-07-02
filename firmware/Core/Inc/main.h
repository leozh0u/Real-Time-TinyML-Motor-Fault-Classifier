/*
 * main.h — placeholder. Populate once STM32CubeMX config is generated.
 *
 * Expected contents once real: peripheral handle externs (SPI, ADC, TIM, USART),
 * fault-class enum (must match training/models label order exactly), and
 * window/sample-buffer size #defines shared with the feature extraction code.
 */

#ifndef MAIN_H
#define MAIN_H

#include "stm32f4xx_hal.h"

/* TODO: fault class enum — keep in lockstep with training/data_prep label encoding */
typedef enum {
    FAULT_HEALTHY = 0,
    FAULT_IMBALANCE,
    FAULT_LOOSENESS,
    FAULT_OVERLOAD,
    FAULT_CLASS_COUNT
} fault_class_t;

/* Sampling window config — must match training/data_prep feature extraction
 * exactly (same window size, same rate) or offline accuracy won't transfer
 * on-device. Treat these as placeholders until step 3 picks real values
 * based on the FFT resolution the fault frequencies actually need. */
#define SAMPLE_RATE_HZ      800
#define WINDOW_SIZE_SAMPLES 256

/* Build mode: 1 while collecting training data (streams every raw sample
 * over UART via UART_SendSample — see uart_stream.h), 0 once a real model
 * exists on-device and step 8 takes over (streams one label per window via
 * UART_SendLabel instead). Currently in data-collection mode by default
 * since no model exists yet. Flip this by hand when step 8 is ready —
 * there's no runtime auto-detection here on purpose, keeps the firmware
 * simple and the two modes clearly separate. */
#define STREAM_RAW_SAMPLES 1

/* TODO(CubeMX): none of the peripheral init below exists yet — this file
 * assumes an .ioc has been generated with:
 *   SPI1  — ADXL345, mode 3 (CPOL=1, CPHA=1), <=5MHz
 *   ADC1  — ACS712, single channel, 12-bit
 *   TIM2  — encoder mode (port from project 1's config, don't re-derive)
 *   TIM3  — periodic interrupt at SAMPLE_RATE_HZ to drive sampling
 *   USART2 — UART telemetry
 * Generate the .ioc first; CubeMX will overwrite the top of main.c between
 * its USER CODE markers, so keep driver/sampling logic below those markers.
 */

void Error_Handler(void);

#endif /* MAIN_H */
