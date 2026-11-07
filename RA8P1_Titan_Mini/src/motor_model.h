/*
 * Copyright (c) 2006-2021, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2026-11-05     HP       the first version
 */
#ifndef SRC_MOTOR_MODEL_H_
#define SRC_MOTOR_MODEL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int motor_model_init(void);

int motor_model_test(void);
unsigned int motor_model_arena_used(void);
int motor_model_run(const uint8_t* rgb, size_t len);

int motor_run_bottom(const uint8_t *image);
int motor_run_front(const uint8_t *image);

#ifdef __cplusplus
}
#endif



#endif /* SRC_MOTOR_MODEL_H_ */
