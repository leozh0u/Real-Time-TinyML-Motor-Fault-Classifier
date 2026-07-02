#include "adxl345.h"

/* --- low-level SPI helpers ------------------------------------------------ */

static void CS_Low(ADXL345_Handle_t *dev)  { HAL_GPIO_WritePin(dev->cs_port, dev->cs_pin, GPIO_PIN_RESET); }
static void CS_High(ADXL345_Handle_t *dev) { HAL_GPIO_WritePin(dev->cs_port, dev->cs_pin, GPIO_PIN_SET); }

static HAL_StatusTypeDef WriteReg(ADXL345_Handle_t *dev, uint8_t reg, uint8_t val)
{
    uint8_t tx[2] = { (uint8_t)(reg & 0x3F), val }; /* write: bit7=0, bit6=0 (single byte) */
    HAL_StatusTypeDef st;
    CS_Low(dev);
    st = HAL_SPI_Transmit(dev->hspi, tx, 2, 10);
    CS_High(dev);
    return st;
}

static HAL_StatusTypeDef ReadRegs(ADXL345_Handle_t *dev, uint8_t reg, uint8_t *buf, uint8_t len)
{
    uint8_t cmd = (uint8_t)((reg & 0x3F) | ADXL345_SPI_READ | (len > 1 ? ADXL345_SPI_MB : 0));
    HAL_StatusTypeDef st;
    CS_Low(dev);
    st = HAL_SPI_Transmit(dev->hspi, &cmd, 1, 10);
    if (st == HAL_OK) {
        st = HAL_SPI_Receive(dev->hspi, buf, len, 10);
    }
    CS_High(dev);
    return st;
}

/* --- public API ------------------------------------------------------------ */

HAL_StatusTypeDef ADXL345_Init(ADXL345_Handle_t *dev, SPI_HandleTypeDef *hspi,
                                GPIO_TypeDef *cs_port, uint16_t cs_pin, uint8_t odr_code)
{
    uint8_t devid = 0;
    HAL_StatusTypeDef st;

    dev->hspi = hspi;
    dev->cs_port = cs_port;
    dev->cs_pin = cs_pin;
    CS_High(dev); /* idle high */

    st = ReadRegs(dev, ADXL345_REG_DEVID, &devid, 1);
    if (st != HAL_OK || devid != ADXL345_DEVID_EXPECTED) {
        return HAL_ERROR;
    }

    st = WriteReg(dev, ADXL345_REG_BW_RATE, odr_code);
    if (st != HAL_OK) return st;

    /* DATA_FORMAT: full resolution (0x08) | +-4g range (0x01) */
    st = WriteReg(dev, ADXL345_REG_DATA_FORMAT, 0x08 | 0x01);
    if (st != HAL_OK) return st;

    /* POWER_CTL: measurement mode */
    return WriteReg(dev, ADXL345_REG_POWER_CTL, 0x08);
}

HAL_StatusTypeDef ADXL345_ReadRaw(ADXL345_Handle_t *dev, ADXL345_Raw_t *out)
{
    uint8_t buf[6];
    HAL_StatusTypeDef st = ReadRegs(dev, ADXL345_REG_DATAX0, buf, 6);
    if (st != HAL_OK) return st;

    out->x = (int16_t)((buf[1] << 8) | buf[0]);
    out->y = (int16_t)((buf[3] << 8) | buf[2]);
    out->z = (int16_t)((buf[5] << 8) | buf[4]);
    return HAL_OK;
}
