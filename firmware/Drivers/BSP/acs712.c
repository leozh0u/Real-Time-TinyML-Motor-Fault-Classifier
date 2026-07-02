#include "acs712.h"

void ACS712_Init(ACS712_Handle_t *dev, ADC_HandleTypeDef *hadc, uint32_t adc_channel, float vref_mv)
{
    dev->hadc = hadc;
    dev->adc_channel = adc_channel;
    dev->vref_mv = vref_mv;
    dev->zero_offset_mv = vref_mv / 2.0f; /* nominal — overwritten by ACS712_Calibrate() */
}

static float ReadRaw_mV(ACS712_Handle_t *dev)
{
    uint32_t raw;
    HAL_ADC_Start(dev->hadc);
    HAL_ADC_PollForConversion(dev->hadc, 10);
    raw = HAL_ADC_GetValue(dev->hadc);
    HAL_ADC_Stop(dev->hadc);
    /* F401 ADC is 12-bit: 0-4095 */
    return ((float)raw / 4095.0f) * dev->vref_mv;
}

void ACS712_Calibrate(ACS712_Handle_t *dev, uint16_t samples)
{
    float sum = 0.0f;
    uint16_t i;
    for (i = 0; i < samples; i++) {
        sum += ReadRaw_mV(dev);
        HAL_Delay(1);
    }
    dev->zero_offset_mv = sum / (float)samples;
}

float ACS712_ReadCurrent_mA(ACS712_Handle_t *dev)
{
    float mv = ReadRaw_mV(dev);
    float delta_mv = mv - dev->zero_offset_mv;
    return (delta_mv / ACS712_MV_PER_AMP) * 1000.0f;
}
