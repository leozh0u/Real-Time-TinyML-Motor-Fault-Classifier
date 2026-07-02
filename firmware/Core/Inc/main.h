/*
 * main.h — placeholder. Populate once STM32CubeMX config is generated.
 *
 * Expected contents once real: peripheral handle externs (SPI, ADC, TIM, USART),
 * fault-class enum (must match training/models label order exactly), and
 * window/sample-buffer size #defines shared with the feature extraction code.
 */

#ifndef MAIN_H
#define MAIN_H

/* TODO: fault class enum — keep in lockstep with training/data_prep label encoding */
typedef enum {
    FAULT_HEALTHY = 0,
    FAULT_IMBALANCE,
    FAULT_LOOSENESS,
    FAULT_OVERLOAD,
    FAULT_CLASS_COUNT
} fault_class_t;

#endif /* MAIN_H */
