#ifndef ENCODER_H
#define ENCODER_H

#include "stm32f4xx_hal.h"
#include <stdint.h>

/*
 * Thin wrapper around a timer configured in Encoder Mode
 * (CubeMX: TIMx -> Combined Channels -> Encoder Mode).
 *
 * This assumes the actual encoder interface (which timer, which pins) was
 * already worked out in the project 1 motor controller firmware. Port that
 * TIM config into this project's CubeMX .ioc rather than re-deriving it —
 * this file only adds what project 1 didn't need: velocity estimation from
 * encoder delta over a fixed sample interval, used here as a feature
 * channel rather than just PID feedback.
 */

typedef struct {
    TIM_HandleTypeDef *htim;
    int32_t             last_count;
    uint32_t             counts_per_rev; /* encoder CPR x gearbox ratio — from motor datasheet */
} Encoder_Handle_t;

void    Encoder_Init(Encoder_Handle_t *enc, TIM_HandleTypeDef *htim, uint32_t counts_per_rev);

/* Signed delta since last call. Relies on the timer counter being 16-bit
 * and the cast to int16_t below to handle wraparound correctly — if the
 * encoder TIM is configured as 32-bit, this needs to change. */
int32_t Encoder_GetDelta(Encoder_Handle_t *enc);

/* Delta converted to RPM given the elapsed time (ms) since the last call.
 * Caller is responsible for calling this at a known, fixed interval. */
float   Encoder_GetRPM(Encoder_Handle_t *enc, uint32_t elapsed_ms);

#endif /* ENCODER_H */
