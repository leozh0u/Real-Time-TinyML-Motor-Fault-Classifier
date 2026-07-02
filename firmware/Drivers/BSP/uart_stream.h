#ifndef UART_STREAM_H
#define UART_STREAM_H

#include "stm32f4xx_hal.h"
#include <stdint.h>

/*
 * Raw sample streaming protocol, used during data collection (pipeline
 * step 2) and any live telemetry later. Not the final classification-label
 * protocol used in step 8 — that's a separate, much lower-rate message
 * (see UART_SendLabel below) since it only fires once per window, not once
 * per sample.
 *
 * Wire format, one packet per sample tick (little-endian):
 *   [0]     0xAA               sync byte
 *   [1..2]  int16  vib_x       raw ADXL345 X-axis LSBs
 *   [3..6]  float32 current_mA
 *   [7]     uint8  checksum    XOR of bytes [1..6]
 *   -------
 *   8 bytes total
 *
 * At SAMPLE_RATE_HZ (800 in main.h) this is 6400 bytes/sec — fits inside
 * the default 115200 baud UART (~11.5 KB/s usable) with margin. If
 * SAMPLE_RATE_HZ is raised later, bump the baud rate in CubeMX
 * accordingly and update PY_BAUD_RATE in training/data_prep/collect_data.py
 * to match — these two are not automatically kept in sync.
 *
 * The Python side (training/data_prep/collect_data.py) must parse this
 * exact struct layout — see PACKET_STRUCT_FMT there.
 */

#define UART_STREAM_SYNC_BYTE 0xAA
#define UART_STREAM_PACKET_LEN 8

void UART_SendSample(UART_HandleTypeDef *huart, int16_t vib_x, float current_mA);

/* Step 8 protocol, once inference exists: one line per completed window,
 * human-readable on purpose so it can be watched directly in a serial
 * monitor (RealTerm, PuTTY, screen) without a decoder — this one is not
 * performance-critical since it fires once per WINDOW_SIZE_SAMPLES, not
 * once per sample. Format: "LABEL,<class_id>,<class_name>\r\n" */
void UART_SendLabel(UART_HandleTypeDef *huart, uint8_t class_id, const char *class_name);

#endif /* UART_STREAM_H */
