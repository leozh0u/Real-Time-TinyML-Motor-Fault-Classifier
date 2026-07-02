#ifndef ACS712_H
#define ACS712_H

#include "stm32f4xx_hal.h"
#include <stdint.h>

/*
 * ACS712-05B current sensor driver, 185 mV/A sensitivity, 0A output = VCC/2.
 *
 * HARDWARE WARNING — check this before wiring the module's OUT pin to the
 * F401's ADC input. Most ACS712 breakout boards run their sensor supply
 * (and therefore their output swing) off 5V, with 0A centered at ~2.5V and
 * swinging +-925mV at the rated +-5A (185mV/A x 5A). That puts the high
 * end of the swing at ~3.425V — above the F401's 3.3V VDDA and outside
 * the ADC pin's safe input range. Sustained high current draw at that
 * point risks clipping the reading at best, damaging the ADC pin at worst.
 * This was not flagged in the original hardware list. Before powering
 * this up, do one of:
 *   (a) confirm your specific breakout can run its sensor supply off 3.3V
 *       (check the regulator on the board, not just the ACS712 datasheet —
 *       the IC itself is rated 3V-5.5V, but breakout boards vary), or
 *   (b) add a resistor divider or op-amp buffer between OUT and the ADC
 *       pin to scale the swing into 0-3.3V.
 */

typedef struct {
    ADC_HandleTypeDef *hadc;
    uint32_t            adc_channel;
    float                vref_mv;         /* ADC reference voltage, mV (e.g. 3300.0f) */
    float                zero_offset_mv;  /* measured 0A output, set by ACS712_Calibrate */
} ACS712_Handle_t;

#define ACS712_MV_PER_AMP 185.0f

void  ACS712_Init(ACS712_Handle_t *dev, ADC_HandleTypeDef *hadc, uint32_t adc_channel, float vref_mv);

/* Call with the motor stopped (zero current flowing) to measure the true
 * zero-current offset. ACS712 modules vary board-to-board and rarely sit
 * exactly at VCC/2 in practice — skipping this biases every reading.
 * Averages `samples` ADC reads with a 1ms delay between them. */
void  ACS712_Calibrate(ACS712_Handle_t *dev, uint16_t samples);

/* Blocking single-conversion read, returns current in mA (signed) */
float ACS712_ReadCurrent_mA(ACS712_Handle_t *dev);

#endif /* ACS712_H */
