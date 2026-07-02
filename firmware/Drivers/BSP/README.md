# firmware/Drivers/BSP/

Hand-written drivers not covered by STM32 HAL.

Expected files once real:
- `adxl345.c` / `adxl345.h` — SPI accelerometer driver, up to 3.2kHz sampling
- `acs712.c` / `acs712.h` — ADC current sensor read + 185mV/A conversion
- `encoder.c` / `encoder.h` — quadrature encoder timer interface (likely shared with project 1's PID controller)
