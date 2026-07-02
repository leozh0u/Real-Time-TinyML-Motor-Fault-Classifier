# docs/

System diagram, wiring notes, and design decision log. Not yet populated. Should eventually include:

- Wiring diagram: ADXL345 (SPI) + ACS712 (ADC) + encoder (timer) + TB6612FNG driver pinout on the F401RE
- System block diagram: sensors -> FFT -> CNN inference -> UART, alongside the PID control loop
- Decision log: why STM32Cube AI Studio vs TFLite Micro (once decided), why these fault classes, why this window size
