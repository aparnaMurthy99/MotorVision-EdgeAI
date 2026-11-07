/*
 * Copyright (c) 2006-2021, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2026-11-05     HP       the first version
 */

#include <rtthread.h>
#include <stdint.h>
#include <stddef.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/activations.h"

#include "motor_model.h"

#include "../model/motor_fusion_int8_vela.h"
#include "tensorflow/lite/micro/kernels/ethosu.h"

extern const unsigned char motor_fusion_int8_vela_tflite[];
extern const unsigned int motor_fusion_int8_vela_tflite_len;

//#define TENSOR_ARENA_SIZE (512U * 1024U)
#define TENSOR_ARENA_SIZE (7U * 1024U * 1024U)

static uint8_t tensor_arena[TENSOR_ARENA_SIZE]
    __attribute__((aligned(16), section(".sdram")));

static tflite::MicroMutableOpResolver<49> resolver;

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* in_t = nullptr;     // set after AllocateTensors(): in_t = interpreter->input(0);
static TfLiteTensor* out_t = nullptr;

//extern volatile int failed_operator_i;
//extern volatile int failed_builtin_code ;
//extern volatile int failed_builtin_version ;
//extern char failed_custom_name[32];

//extern volatile int dbg_stage;
//extern volatile int dbg_node;
//extern volatile int dbg_code;
//extern volatile int dbg_prepare_status ;
//extern volatile int dbg_ethos_line ;
//extern volatile int dbg_alloc_step ;

volatile int8_t *debug_input_array = nullptr;
volatile int debug_input_bytes = 0;
volatile int8_t debug_samples[3];

extern "C" int motor_model_init(void)
{
    const tflite::Model *model =
        tflite::GetModel(motor_model_data);

    if (model == nullptr)
        return -1;

    if (model->version() != TFLITE_SCHEMA_VERSION)
        return -2;

    // Register ALL operators first
    if (resolver.AddRelu0To1() != kTfLiteOk)
        return -3;

    if (resolver.AddDequantize() != kTfLiteOk)
        return -4;

    if (resolver.AddShape() != kTfLiteOk)
        return -5;

    if (resolver.AddFill() != kTfLiteOk)
        return -6;

    if (resolver.AddQuantize() != kTfLiteOk)
        return -7;

    if (resolver.AddMaximum() != kTfLiteOk)
        return -8;

    if (resolver.AddSlice() != kTfLiteOk)
        return -9;

    if (resolver.AddMean() != kTfLiteOk)
        return -10;

    if (resolver.AddCustom(
            "ethos-u",
            tflite::Register_ETHOSU()) != kTfLiteOk)
        return -11;

    // Create interpreter
    interpreter = new tflite::MicroInterpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );

    if (interpreter == nullptr)
        return -12;

    // Allocate tensors only after registration
    TfLiteStatus status = interpreter->AllocateTensors();

    in_t = interpreter->input(0);

    if (in_t == nullptr)
        return -14;

    if (status != kTfLiteOk)
        return -13;

    return 0;
}


extern "C" int motor_model_run(const uint8_t* rgb, size_t len)
{
    if (in_t == nullptr) return -40;
    if (len != in_t->bytes) return -41;

    int8_t lut[256];
    const float s = in_t->params.scale;
    const int zp = in_t->params.zero_point;

    for (int p = 0; p < 256; p++)
    {
        int q = (int)lroundf((p / 255.0f) / s) + zp;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        lut[p] = (int8_t)q;
    }

    for (size_t i = 0; i < len; i++)
        in_t->data.int8[i] = lut[rgb[i]];

    rt_kprintf("ACQUIRED RGB: %u %u %u\n",
               rgb[0], rgb[1], rgb[2]);

    rt_kprintf("MODEL INPUT: %d %d %d\n",
               in_t->data.int8[0],
               in_t->data.int8[1],
               in_t->data.int8[2]);


    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
        return -42;

    // Read output
    out_t = interpreter->output(0);

    int8_t score_0 = out_t->data.int8[0];  // NO FAULT
    int8_t score_1 = out_t->data.int8[1];  // FAULT

    float prob_0 = (score_0 - out_t->params.zero_point)
                   * out_t->params.scale;

    float prob_1 = (score_1 - out_t->params.zero_point)
                   * out_t->params.scale;

    rt_kprintf("NO FAULT: %.4f\n", prob_0);
    rt_kprintf("FAULT: %.4f\n", prob_1);

    // Model classification 
    return (score_1 > score_0) ? 1 : 0;


}

extern "C" unsigned int motor_model_arena_used(void)
{
    volatile int model_input_count = interpreter->inputs_size();
    return (unsigned int)interpreter->arena_used_bytes();
}


extern "C" int motor_model_test(void)
{
    // 1. Get input tensor
    TfLiteTensor* in = interpreter->input(0);
    if (in == nullptr) return -30;

    // Inspect geometry input (input 1)
    TfLiteTensor* geo_in = interpreter->input(1);
    if (geo_in == nullptr) return -33;

    volatile int geo_d0 = geo_in->dims->data[0];
    volatile int geo_d1 = geo_in->dims->data[1];
    volatile int geo_type = (int)geo_in->type;
    volatile int geo_bytes = geo_in->bytes;
    volatile float geo_scale = geo_in->params.scale;
    volatile int geo_zero_point = geo_in->params.zero_point;

    // Original training geometry features:
    // [OD_Flange, OD_Gear, GearCount]

    const float geo_values[3] = {
        99.44135f,   // Temporary: training mean
        22.018229f,  // Temporary: training mean
        15.0f        // Temporary: training mean
    };

    for (int i = 0; i < 3; i++)
    {
        geo_in->data.int8[i] = (int8_t)lroundf(
            geo_values[i] / geo_in->params.scale
            + geo_in->params.zero_point
        );
    }

    // 2. Inspect input tensor
    volatile int input_d0 = in->dims->data[0];
    volatile int input_d1 = in->dims->data[1];
    volatile int input_d2 = in->dims->data[2];
    volatile int input_d3 = in->dims->data[3];

    volatile int input_type = (int)in->type;
    volatile int input_bytes = in->bytes;
    volatile float input_scale = in->params.scale;
    volatile int input_zero_point = in->params.zero_point;

    // 3. Temporary test pattern: horizontal gradient
    const int H = input_d1;
    const int W = input_d2;

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            uint8_t p = (uint8_t)((x * 255) / (W - 1));

            for (int c = 0; c < 3; c++)
            {
                int idx = (y * W + x) * 3 + c;

                in->data.int8[idx] =
                    (int8_t)(lroundf((p / 255.0f) / input_scale)
                             + input_zero_point);
            }
        }
    }

    // 4. Debug input
    debug_input_array = in->data.int8;
    debug_input_bytes = in->bytes;

    debug_samples[0] = in->data.int8[0];
    debug_samples[1] = in->data.int8[100];
    debug_samples[2] = in->data.int8[1000];

    // 5. Run inference
    if (interpreter->Invoke() != kTfLiteOk)
        return -31;

    // 6. Get output tensor
    TfLiteTensor* out = interpreter->output(0);
    if (out == nullptr) return -32;

    // 7. Inspect output tensor

    volatile int8_t result_0 = out->data.int8[0];
    volatile int8_t result_1 = out->data.int8[1];

    volatile float score_0 =
        (result_0 - 20) * 0.003107f;

    volatile float score_1 =
        (result_1 - 20) * 0.003107f;

//    volatile int output_bytes = out->bytes;
//    volatile int output_type = (int)out->type;

    // 8. Inspect affine quantization
    if (out->quantization.type == kTfLiteAffineQuantization &&
        out->quantization.params != nullptr)
    {
        TfLiteAffineQuantization* affine =
            (TfLiteAffineQuantization*)out->quantization.params;

        if (affine->scale != nullptr && affine->scale->size > 0)
        {
            volatile int affine_scale_size = affine->scale->size;
            volatile float actual_scale = affine->scale->data[0];
        }

        if (affine->zero_point != nullptr &&
            affine->zero_point->size > 0)
        {
            volatile int affine_zp_size = affine->zero_point->size;
            volatile int actual_zero_point =
                affine->zero_point->data[0];
        }
    }

    return 0;
}
