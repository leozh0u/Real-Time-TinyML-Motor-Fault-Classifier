#include "uart_stream.h"
#include <string.h>
#include <stdio.h>

void UART_SendSample(UART_HandleTypeDef *huart, int16_t vib_x, float current_mA)
{
    uint8_t pkt[UART_STREAM_PACKET_LEN];
    uint8_t checksum = 0;
    int i;

    pkt[0] = UART_STREAM_SYNC_BYTE;
    memcpy(&pkt[1], &vib_x, sizeof(int16_t));
    memcpy(&pkt[3], &current_mA, sizeof(float));

    for (i = 1; i <= 6; i++) {
        checksum ^= pkt[i];
    }
    pkt[7] = checksum;

    /* Blocking transmit — fine at 800Hz/8 bytes on a UART running well
     * above that throughput. If this ever needs to run concurrently with
     * tight PID timing, switch to HAL_UART_Transmit_DMA instead. */
    HAL_UART_Transmit(huart, pkt, UART_STREAM_PACKET_LEN, 10);
}

void UART_SendLabel(UART_HandleTypeDef *huart, uint8_t class_id, const char *class_name)
{
    char line[64];
    int len = snprintf(line, sizeof(line), "LABEL,%u,%s\r\n", (unsigned)class_id, class_name);
    if (len > 0) {
        HAL_UART_Transmit(huart, (uint8_t *)line, (uint16_t)len, 50);
    }
}
