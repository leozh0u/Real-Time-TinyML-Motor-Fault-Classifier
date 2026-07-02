#include "encoder.h"

void Encoder_Init(Encoder_Handle_t *enc, TIM_HandleTypeDef *htim, uint32_t counts_per_rev)
{
    enc->htim = htim;
    enc->counts_per_rev = counts_per_rev;
    HAL_TIM_Encoder_Start(htim, TIM_CHANNEL_ALL);
    enc->last_count = (int16_t)__HAL_TIM_GET_COUNTER(htim);
}

int32_t Encoder_GetDelta(Encoder_Handle_t *enc)
{
    int16_t now = (int16_t)__HAL_TIM_GET_COUNTER(enc->htim); /* signed 16-bit cast handles wraparound */
    int32_t delta = (int32_t)now - enc->last_count;
    enc->last_count = now;
    return delta;
}

float Encoder_GetRPM(Encoder_Handle_t *enc, uint32_t elapsed_ms)
{
    int32_t delta = Encoder_GetDelta(enc);
    float revs;
    if (elapsed_ms == 0 || enc->counts_per_rev == 0) return 0.0f;
    revs = (float)delta / (float)enc->counts_per_rev;
    return revs * (60000.0f / (float)elapsed_ms);
}
