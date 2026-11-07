#include <Arduino.h>
#include "driver/spi_slave.h"
#include "esp_heap_caps.h"
#include "esp_camera.h"
#include "img_converters.h"

// ---------------- Camera pins ----------------
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5

#define CAM_PIN_D0     11
#define CAM_PIN_D1      9
#define CAM_PIN_D2      8
#define CAM_PIN_D3     10
#define CAM_PIN_D4     12
#define CAM_PIN_D5     18
#define CAM_PIN_D6     17
#define CAM_PIN_D7     16

#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK   13

// ---------------- SPI slave pins (Titan is master) ----------------
#define PIN_SCLK       39     // <- Titan P102
#define PIN_CS         40     // <- Titan P105
#define PIN_MOSI       41     // <- Titan P708
#define PIN_MISO       42     // -> Titan P709

#define CAPTURE_REQ_PIN 14    // <- Titan capture request

// ---------------- Image sizes ----------------
#define SRC_W 1280
#define SRC_H 1024
#define RGB_SIZE (SRC_W * SRC_H * 3)

#define OUT_W 480
#define OUT_H 384
#define OUT_SIZE (OUT_W * OUT_H * 3)

static constexpr size_t FRAME_SIZE = OUT_SIZE;               // 552960
static constexpr size_t CHUNK_SIZE = 12288;
static constexpr size_t CHUNKS_PER_FRAME = FRAME_SIZE / CHUNK_SIZE;   // 45
static_assert(FRAME_SIZE % CHUNK_SIZE == 0, "Frame size must be divisible by chunk size");

WORD_ALIGNED_ATTR static uint8_t txbuf[CHUNK_SIZE];
WORD_ALIGNED_ATTR static uint8_t hdrbuf[16];

static bool g_ready = false;

// The first CS falling edge from the Titan (it starts polling right after the
// user button is pressed) is the capture trigger. No separate trigger wire.
static volatile bool g_trig = false;
static void IRAM_ATTR isrCsFall() { g_trig = true; }

// Used on every exit path: let the Titan's polling settle, then re-arm trigger
static void waitRelease()
{
    delay(300);
    g_trig = false;
}

void setup()
{
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== ESP32-S3 SPI slave: sends frame ===");
    Serial.flush();

    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;

    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;

    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;

    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;

    config.xclk_freq_hz = 20000000;

    // Capture JPEG; convert to RGB888 on ESP32
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_SXGA;
    config.jpeg_quality = 12;

    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    pinMode(CAPTURE_REQ_PIN, INPUT_PULLDOWN);

    spi_bus_config_t bus = {};
    bus.mosi_io_num     = PIN_MOSI;
    bus.miso_io_num     = PIN_MISO;
    bus.sclk_io_num     = PIN_SCLK;
    bus.quadwp_io_num   = -1;
    bus.quadhd_io_num   = -1;
    bus.max_transfer_sz = CHUNK_SIZE;

    spi_slave_interface_config_t slv = {};
    slv.spics_io_num = PIN_CS;
    slv.mode         = 0;
    slv.queue_size   = 1;
    slv.flags        = 0;

    esp_err_t e = spi_slave_initialize(SPI2_HOST, &bus, &slv, SPI_DMA_CH_AUTO);
    Serial.printf("slave init: %d (0 = OK)\n", (int)e);
    if (e != ESP_OK) return;

    Serial.println("Initializing camera...");
    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("CAMERA_ERROR");
        return;
    }

    // Discard the first frames (auto-exposure settles)
    for (int i = 0; i < 3; i++) {
        camera_fb_t *w = esp_camera_fb_get();
        if (w) esp_camera_fb_return(w);
    }

    g_trig = false;
    attachInterrupt(PIN_CS, isrCsFall, FALLING);   // CS pulse = trigger

    g_ready = true;
    Serial.println("READY");
    Serial.flush();
}

void loop()
{
    if (!g_ready) {
        delay(1000);
        return;
    }

    // Wait for the first CS pulse from the Titan
    while (!g_trig) {
        delay(1);
    }

    Serial.println("TITAN TRIGGER RECEIVED (CS)");
    Serial.flush();

    // 1. Capture JPEG
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == nullptr) {
        Serial.println("CAPTURE_ERROR");
        waitRelease();
        return;
    }

    if (fb->width != SRC_W || fb->height != SRC_H) {
        esp_camera_fb_return(fb);
        Serial.println("RESOLUTION_ERROR");
        waitRelease();
        return;
    }

    // 2. Allocate full-resolution RGB888 in PSRAM
    uint8_t *rgb = (uint8_t *)heap_caps_malloc(RGB_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (rgb == nullptr) {
        esp_camera_fb_return(fb);
        Serial.println("MEMORY_ERROR");
        waitRelease();
        return;
    }

    // 3. Decode JPEG -> RGB888 (note: byte order is B,G,R)
    bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb);
    esp_camera_fb_return(fb);

    if (!ok) {
        heap_caps_free(rgb);
        Serial.println("CONVERSION_ERROR");
        waitRelease();
        return;
    }

    // 4. Allocate final output buffer
    uint8_t *output = (uint8_t *)heap_caps_malloc(OUT_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (output == nullptr) {
        heap_caps_free(rgb);
        Serial.println("OUTPUT_MEMORY_ERROR");
        waitRelease();
        return;
    }

    const uint32_t stepX = ((uint32_t)SRC_W << 16) / OUT_W;
    const uint32_t stepY = ((uint32_t)SRC_H << 16) / OUT_H;

    for (int y = 0; y < OUT_H; y++) {
        uint32_t srcY = (y * stepY) >> 16;
        int srcRowOffset = srcY * SRC_W;
        int dstRowOffset = y * OUT_W;

        for (int x = 0; x < OUT_W; x++) {
            uint32_t srcX = (x * stepX) >> 16;
            int srcIndex = (srcRowOffset + srcX) * 3;
            int dstIndex = (dstRowOffset + x) * 3;

            output[dstIndex]     = rgb[srcIndex];
            output[dstIndex + 1] = rgb[srcIndex + 1];
            output[dstIndex + 2] = rgb[srcIndex + 2];
        }
    }

    heap_caps_free(rgb);

    Serial.printf("IMG:%u:%u:%u:RGB888\n", (unsigned int)OUT_SIZE, OUT_W, OUT_H);
    Serial.flush();

    // 5a. Handshake: 16-byte header with magic bytes, sent once per frame
    memset(hdrbuf, 0, sizeof(hdrbuf));
    hdrbuf[0] = 0xA5; hdrbuf[1] = 0x5A; hdrbuf[2] = 0xC3; hdrbuf[3] = 0x3C;

    spi_slave_transaction_t h = {};
    h.length    = sizeof(hdrbuf) * 8;
    h.tx_buffer = hdrbuf;
    h.rx_buffer = nullptr;

    esp_err_t he = spi_slave_transmit(SPI2_HOST, &h, pdMS_TO_TICKS(10000));
    if (he != ESP_OK) {
        Serial.printf("HEADER error: %d\n", (int)he);
        heap_caps_free(output);
        waitRelease();
        return;
    }
    Serial.println("HEADER sent");

    // 5b. Send the 45 image chunks
    for (size_t chunk = 0; chunk < CHUNKS_PER_FRAME; chunk++)
    {
        memcpy(txbuf, output + (chunk * CHUNK_SIZE), CHUNK_SIZE);

        spi_slave_transaction_t t = {};
        t.length    = CHUNK_SIZE * 8;
        t.tx_buffer = txbuf;
        t.rx_buffer = nullptr;

        esp_err_t e = spi_slave_transmit(SPI2_HOST, &t, pdMS_TO_TICKS(2000));
        if (e != ESP_OK) {
            Serial.printf("SPI error at chunk %u: %d\n", (unsigned)(chunk + 1), (int)e);
            break;
        }
    }

    heap_caps_free(output);
    Serial.println("FRAME DONE");
    waitRelease();
}