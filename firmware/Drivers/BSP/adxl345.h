#ifndef ADXL345_H
#define ADXL345_H

#include "stm32f4xx_hal.h"
#include <stdint.h>

/*
 * ADXL345 accelerometer driver, SPI mode.
 *
 * WIRING NOTE: ADXL345 requires SPI mode 3 (CPOL=1, CPHA=1). If CubeMX's
 * SPI1 config defaults to mode 0, reads will come back garbage or DEVID
 * won't match — this is the most common ADXL345 SPI bring-up failure.
 * Max SPI clock is 5 MHz per datasheet.
 */

#define ADXL345_REG_DEVID       0x00
#define ADXL345_REG_BW_RATE     0x2C
#define ADXL345_REG_POWER_CTL   0x2D
#define ADXL345_REG_DATA_FORMAT 0x31
#define ADXL345_REG_DATAX0      0x32
#define ADXL345_DEVID_EXPECTED  0xE5

#define ADXL345_SPI_READ 0x80
#define ADXL345_SPI_MB   0x40  /* multi-byte transfer */

/* BW_RATE output data rate codes — see datasheet Table 7 */
#define ADXL345_RATE_3200HZ 0x0F
#define ADXL345_RATE_1600HZ 0x0E
#define ADXL345_RATE_800HZ  0x0D
#define ADXL345_RATE_400HZ  0x0C

typedef struct {
    SPI_HandleTypeDef *hspi;
    GPIO_TypeDef       *cs_port;
    uint16_t            cs_pin;
} ADXL345_Handle_t;

typedef struct {
    int16_t x;
    int16_t y;
    int16_t z;
} ADXL345_Raw_t;

/* Returns HAL_OK only if DEVID reads back as 0xE5. Returning HAL_ERROR here
 * almost always means wrong SPI mode, wrong CS pin, or a wiring fault —
 * don't proceed past a failed init in the real build. */
HAL_StatusTypeDef ADXL345_Init(ADXL345_Handle_t *dev, SPI_HandleTypeDef *hspi,
                                GPIO_TypeDef *cs_port, uint16_t cs_pin, uint8_t odr_code);

/* Blocking read of one X/Y/Z sample (full-resolution mode, +-4g range) */
HAL_StatusTypeDef ADXL345_ReadRaw(ADXL345_Handle_t *dev, ADXL345_Raw_t *out);

#endif /* ADXL345_H */
