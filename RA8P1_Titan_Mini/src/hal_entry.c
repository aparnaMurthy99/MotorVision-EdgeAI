#include "hal_data.h"
#include <rtthread.h>
#include <rtdevice.h>
#include <string.h>
#include <stdbool.h>

#include "motor_model.h"

/* Titan Mini RA8P1 SPI1 pins */
#define PIN_SCLK  BSP_IO_PORT_01_PIN_02   /* P102 */
#define PIN_CS    BSP_IO_PORT_01_PIN_05   /* P105 */
#define PIN_MOSI  BSP_IO_PORT_07_PIN_08   /* P708 */
#define PIN_MISO  BSP_IO_PORT_07_PIN_09   /* P709 */

#define LED_R  BSP_IO_PORT_01_PIN_09
#define LED_G  BSP_IO_PORT_01_PIN_08
#define LED_B  BSP_IO_PORT_01_PIN_10

#define PIN_USER  BSP_IO_PORT_02_PIN_01   /*usr button */

#define PIN_CAPTURE_REQ  BSP_IO_PORT_00_PIN_05

#define SPI_PIN_CFG (IOPORT_CFG_PERIPHERAL_PIN | IOPORT_PERIPHERAL_SPI)

//#define FRAME  12288  // 196608 bytes
//
//static uint8_t tx_dummy[FRAME] __attribute__((aligned(4)));
//static uint8_t rx_buf[FRAME]   __attribute__((aligned(4)));

#define SCLK_HZ   10000000U                 // 1000000U

#define CHUNK_SIZE   (64U * 64U * 3U)
#define IMAGE_SIZE   (480U * 384U * 3U)
#define NUM_CHUNKS   45U
////#define NUM_FRAMES   2U
//
#define IMAGE_WIDTH   480U
#define IMAGE_HEIGHT  384U
#define CHANNELS      3U
#define PIXEL_SIZE    3U
#define ROW_SIZE      (IMAGE_WIDTH * CHANNELS)


/* DMA-aligned buffers */
static uint8_t tx_dummy[CHUNK_SIZE] __attribute__((aligned(4)));

//static uint8_t rx_frames[NUM_FRAMES][IMAGE_SIZE]
//    __attribute__((aligned(4)));

static uint8_t rx_frames[IMAGE_SIZE]
    __attribute__((aligned(4)));

extern volatile int8_t *debug_input_array;
extern volatile int debug_input_bytes;


/* Debug variables */
volatile uint8_t g_first = 0;
volatile uint8_t g_second = 0;
volatile uint8_t g_third = 0;
volatile uint8_t g_last = 0;

volatile uint32_t g_open_err = 0;
volatile uint32_t g_div_err = 0;
volatile uint32_t g_xfer_err = 0;
volatile uint32_t g_done = 0;
volatile uint32_t g_spi_errs = 0;
volatile uint32_t g_frames = 0;
volatile uint32_t g_timeout = 0;

volatile uint32_t g_mismatch = 0xFFFFFFFFU;
volatile uint32_t g_first_bad = 0xFFFFFFFFU;

bsp_io_level_t button_level;
uint8_t valid = 0;

void set_led(uint8_t r, uint8_t g, uint8_t b)
{
    R_IOPORT_PinWrite(&g_ioport_ctrl, LED_R,
        r ? BSP_IO_LEVEL_LOW : BSP_IO_LEVEL_HIGH);

    R_IOPORT_PinWrite(&g_ioport_ctrl, LED_G,
        g ? BSP_IO_LEVEL_LOW : BSP_IO_LEVEL_HIGH);

    R_IOPORT_PinWrite(&g_ioport_ctrl, LED_B,
        b ? BSP_IO_LEVEL_LOW : BSP_IO_LEVEL_HIGH);
}

/* SPI callback */
void spi1_callback(spi_callback_args_t *p_args)
{
    if (p_args->event == SPI_EVENT_TRANSFER_COMPLETE)
    {
        g_done = 1;
    }
    else
    {
        g_spi_errs++;
    }
}


/* Configure SPI1 as MASTER at 10 MHz */
static fsp_err_t open_master_10mhz(void)
{
    static spi_cfg_t cfg;
    static spi_b_extended_cfg_t ext;

    ext = *(spi_b_extended_cfg_t *)g_spi1.p_cfg->p_extend;

    g_div_err = (uint32_t)
        R_SPI_B_CalculateBitrate(
            SCLK_HZ,
            ext.clock_source,
            &ext.spck_div
        );

    if (g_div_err != FSP_SUCCESS)
    {
        return (fsp_err_t)g_div_err;
    }

    cfg = *g_spi1.p_cfg;

    cfg.operating_mode = SPI_MODE_MASTER;
    cfg.clk_polarity   = SPI_CLK_POLARITY_LOW;
    cfg.clk_phase      = SPI_CLK_PHASE_EDGE_ODD;
    cfg.bit_order      = SPI_BIT_ORDER_MSB_FIRST;

    cfg.p_callback = spi1_callback;
    cfg.p_context  = NULL;
    cfg.p_extend   = &ext;

    return g_spi1.p_api->open(g_spi1.p_ctrl, &cfg);
}

static int validate_frame(const uint8_t *buffer, size_t len)
{
    if (buffer == NULL || len == 0)
        return 0;

    int all_zero = 1;
    int all_255 = 1;

    for (size_t i = 0; i < len; i++)
    {
        if (buffer[i] != 0)
            all_zero = 0;

        if (buffer[i] != 255)
            all_255 = 0;

        if (!all_zero && !all_255)
            return 1;
    }

    return 0;
}

#define HDR_LEN        16U
#define HDR_POLL_MAX   2000U      /* ~20-40 s max wait for the ESP32 */

static uint8_t hdr_rx[32] __attribute__((aligned(32)));
static uint8_t g_hs_diag = 0;     /* 1 = xfer error, 2 = all 00, 3 = all FF, 4 = other data */

/* One CS-framed transfer. Returns 1 on success, 0 on error or timeout. */
static int spi_xfer(uint8_t *rx, uint32_t len, uint32_t timeout_ms)
{
    g_done = 0;

    R_IOPORT_PinWrite(&g_ioport_ctrl, PIN_CS, BSP_IO_LEVEL_LOW);

    g_xfer_err = (uint32_t)g_spi1.p_api->writeRead(
        g_spi1.p_ctrl, tx_dummy, rx, len, SPI_BIT_WIDTH_8_BITS);

    if (g_xfer_err != FSP_SUCCESS)
    {
        R_IOPORT_PinWrite(&g_ioport_ctrl, PIN_CS, BSP_IO_LEVEL_HIGH);
        return 0;
    }

    while (!g_done && timeout_ms > 0U)
    {
        rt_thread_mdelay(1);
        timeout_ms--;
    }

    R_IOPORT_PinWrite(&g_ioport_ctrl, PIN_CS, BSP_IO_LEVEL_HIGH);

    if (!g_done)
    {
        g_timeout++;
        return 0;
    }

    SCB_InvalidateDCache_by_Addr(rx, (int32_t)len);
    return 1;
}

/* Poll the 16-byte header until the ESP32 answers with the magic bytes.
 * The first poll's CS pulse is also what triggers the ESP32 capture. */
static int wait_esp_ready(void)
{
    g_hs_diag = 0;

    for (uint32_t tries = 0; tries < HDR_POLL_MAX; tries++)
    {
        memset(hdr_rx, 0, sizeof(hdr_rx));

        if (!spi_xfer(hdr_rx, HDR_LEN, 20U))
        {
            if (g_hs_diag < 1) g_hs_diag = 1;
        }
        else if (hdr_rx[0] == 0xA5 && hdr_rx[1] == 0x5A &&
                 hdr_rx[2] == 0xC3 && hdr_rx[3] == 0x3C)
        {
            return 1;
        }
        else if (hdr_rx[0] == 0x00 && hdr_rx[1] == 0x00 &&
                 hdr_rx[2] == 0x00 && hdr_rx[3] == 0x00)
        {
            if (g_hs_diag < 2) g_hs_diag = 2;
        }
        else if (hdr_rx[0] == 0xFF && hdr_rx[1] == 0xFF &&
                 hdr_rx[2] == 0xFF && hdr_rx[3] == 0xFF)
        {
            if (g_hs_diag < 3) g_hs_diag = 3;
        }
        else
        {
            g_hs_diag = 4;
        }

        rt_thread_mdelay(10);
    }
    return 0;
}

/* Error: show red, print the reason, then go back to READY */
static void fail(const char *msg)
{
    set_led(1, 0, 0);
    rt_kprintf("ERROR: %s (spi_errs=%u timeouts=%u xfer_err=%u)\n",
               msg, (unsigned)g_spi_errs, (unsigned)g_timeout, (unsigned)g_xfer_err);
    rt_thread_mdelay(3000);
}

/* hal_entry with capture + handshake + motor model.
 * Assumes the helpers from the working capture version are still defined above it:
 *   spi_xfer(), wait_esp_ready(), g_hs_diag, fail(), validate_frame(), set_led()
 * and that the Ethos-U / OSPI / model globals from your motor project exist.
 */
void hal_entry(void)
{
    memset(tx_dummy, 0x00, sizeof(tx_dummy));
    memset(rx_frames, 0x00, sizeof(rx_frames));

    g_spi1.p_api->close(g_spi1.p_ctrl);

    g_open_err = (uint32_t)open_master_10mhz();
    if (g_open_err != FSP_SUCCESS)
    {
        while (1)
        {
            rt_thread_mdelay(1000);
        }
    }

    R_IOPORT_PinCfg(&g_ioport_ctrl, LED_R, IOPORT_CFG_PORT_DIRECTION_OUTPUT | IOPORT_CFG_PORT_OUTPUT_HIGH);
    R_IOPORT_PinCfg(&g_ioport_ctrl, LED_G, IOPORT_CFG_PORT_DIRECTION_OUTPUT | IOPORT_CFG_PORT_OUTPUT_HIGH);
    R_IOPORT_PinCfg(&g_ioport_ctrl, LED_B, IOPORT_CFG_PORT_DIRECTION_OUTPUT | IOPORT_CFG_PORT_OUTPUT_HIGH);

    R_IOPORT_PinCfg(&g_ioport_ctrl, PIN_SCLK, SPI_PIN_CFG);
    R_IOPORT_PinCfg(&g_ioport_ctrl, PIN_MOSI, SPI_PIN_CFG);
    R_IOPORT_PinCfg(&g_ioport_ctrl, PIN_MISO, SPI_PIN_CFG);

    R_IOPORT_PinCfg(&g_ioport_ctrl, PIN_USER, IOPORT_CFG_PORT_DIRECTION_INPUT);

    R_IOPORT_PinCfg(&g_ioport_ctrl, PIN_CS,
                    IOPORT_CFG_PORT_DIRECTION_OUTPUT | IOPORT_CFG_PORT_OUTPUT_HIGH);

    rt_thread_mdelay(100);

    /* Open Ethos-U */
    fsp_err_t ethos_err = g_rm_ethosu0.p_api->open(&g_rm_ethosu0_ctrl, &g_rm_ethosu0_cfg);
    if (ethos_err != FSP_SUCCESS)
    {
        rt_kprintf("Ethos-U open failed: %d\n", ethos_err);
        set_led(1, 0, 0);
        while (1) rt_thread_mdelay(1000);
    }

    /* Open OSPI */
    fsp_err_t ospi_err = R_OSPI_B_Open(&g_ospi_b_ctrl, &g_ospi_b_cfg);
    if (ospi_err != FSP_SUCCESS)
    {
        rt_kprintf("OSPI open failed: %d\n", ospi_err);
        set_led(1, 0, 0);
        while (1) rt_thread_mdelay(1000);
    }

    /* Initialize TFLite model */
    int model_status = motor_model_init();
    if (model_status != 0)
    {
        rt_kprintf("Motor model init failed: %d\n", model_status);
        set_led(1, 0, 0);
        while (1) rt_thread_mdelay(1000);
    }
    rt_kprintf("Motor model initialized successfully\n");

    while (1)
    {
        /* ---- Wait for button press ---- */
        set_led(1, 1, 1);   /* WHITE: READY */
        do
        {
            R_IOPORT_PinRead(&g_ioport_ctrl, PIN_USER, &button_level);
            rt_thread_mdelay(10);
        } while (button_level != BSP_IO_LEVEL_LOW);

        rt_thread_mdelay(30);   /* debounce */
        R_IOPORT_PinRead(&g_ioport_ctrl, PIN_USER, &button_level);
        if (button_level != BSP_IO_LEVEL_LOW)
        {
            continue;           /* bounce, not a real press */
        }

        set_led(0, 0, 1);       /* BLUE: COLLECTION */

        /* ---- Handshake: first CS pulse triggers the ESP32 capture; keep polling
         *      until the ESP32 answers with the magic header ---- */
        if (!wait_esp_ready())
        {
            rt_kprintf("handshake timeout, diag=%u\n", (unsigned)g_hs_diag);
            fail("ESP32 handshake timeout");
            continue;
        }

        rt_thread_mdelay(2);    /* let ESP32 copy chunk 0 and arm it */

        /* ---- Receive all chunks ---- */
        int ok = 1;
        for (uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++)
        {
            if (!spi_xfer(&rx_frames[chunk * CHUNK_SIZE], CHUNK_SIZE, 100U))
            {
                rt_kprintf("chunk %u failed\n", (unsigned)chunk);
                ok = 0;
                break;
            }
            rt_thread_mdelay(2);    /* gap so ESP32 can re-arm next chunk */
        }

        if (!ok)
        {
            fail("SPI chunk transfer");
            continue;
        }

        set_led(0, 1, 0);       /* GREEN: FRAME DONE */
        rt_thread_mdelay(500);

        g_first  = rx_frames[0];
        g_second = rx_frames[1];
        g_third  = rx_frames[2];
        g_last   = rx_frames[IMAGE_SIZE - 1U];

        if (!validate_frame(rx_frames, IMAGE_SIZE))
        {
            fail("frame all 0x00 or all 0xFF");
            continue;
        }

        /* ---- Run the model on the received frame ---- */
        set_led(0, 1, 1);       /* CYAN: RUNNING MODEL */

        rt_tick_t t0 = rt_tick_get();
        int rs = motor_model_run(rx_frames, IMAGE_SIZE);
        rt_tick_t dt = rt_tick_get() - t0;
        rt_kprintf("inference took %u ms\n", (unsigned)(dt * 1000U / RT_TICK_PER_SECOND));

        if (rs == 0)
        {
            set_led(1, 1, 0);   /* YELLOW: NO FAULT */
            rt_kprintf("NO FAULT\n");
        }
        else if (rs == 1)
        {
            set_led(1, 0, 0);   /* RED: FAULT */
            rt_kprintf("FAULT\n");
        }
        else
        {
            rt_kprintf("Model run failed: %d\n", rs);
            set_led(1, 0, 0);
            rt_thread_mdelay(3000);
            continue;           /* don't hang; go back to READY */
        }

        rt_thread_mdelay(1000);
    }
}



//void hal_entry(void)
//{
//    memset(tx_dummy, 0x00, CHUNK_SIZE);
//    memset(rx_frames, 0x00, sizeof(rx_frames));
//
//    g_spi1.p_api->close(g_spi1.p_ctrl);
//
//    g_open_err = (uint32_t)open_master_10mhz();
//
//    if (g_open_err != FSP_SUCCESS)
//    {
//        while (1)
//        {
//            rt_thread_mdelay(1000);
//        }
//    }
//
//    R_IOPORT_PinCfg(&g_ioport_ctrl, LED_R,
//        IOPORT_CFG_PORT_DIRECTION_OUTPUT |
//        IOPORT_CFG_PORT_OUTPUT_HIGH);
//
//    R_IOPORT_PinCfg(&g_ioport_ctrl, LED_G,
//        IOPORT_CFG_PORT_DIRECTION_OUTPUT |
//        IOPORT_CFG_PORT_OUTPUT_HIGH);
//
//    R_IOPORT_PinCfg(&g_ioport_ctrl, LED_B,
//        IOPORT_CFG_PORT_DIRECTION_OUTPUT |
//        IOPORT_CFG_PORT_OUTPUT_HIGH);
//
//    R_IOPORT_PinCfg(
//        &g_ioport_ctrl, PIN_SCLK, SPI_PIN_CFG
//    );
//
//    R_IOPORT_PinCfg(
//        &g_ioport_ctrl, PIN_MOSI, SPI_PIN_CFG
//    );
//
//    R_IOPORT_PinCfg(
//        &g_ioport_ctrl, PIN_MISO, SPI_PIN_CFG
//    );
//
//    R_IOPORT_PinCfg(
//        &g_ioport_ctrl,
//        PIN_USER,
//        IOPORT_CFG_PORT_DIRECTION_INPUT
//    );
//
//    R_IOPORT_PinCfg(
//        &g_ioport_ctrl,
//        PIN_CS,
//        IOPORT_CFG_PORT_DIRECTION_OUTPUT |
//        IOPORT_CFG_PORT_OUTPUT_HIGH
//    );
//
//    rt_thread_mdelay(100);
//    // Open Ethos-U
//    fsp_err_t ethos_err = g_rm_ethosu0.p_api->open(
//        &g_rm_ethosu0_ctrl,
//        &g_rm_ethosu0_cfg
//    );
//
//    if (ethos_err != FSP_SUCCESS)
//    {
//        rt_kprintf("Ethos-U open failed: %d\n", ethos_err);
//        set_led(1, 0, 0);
//        while (1) rt_thread_mdelay(1000);
//    }
//
//    // Open OSPI
//    fsp_err_t ospi_err = R_OSPI_B_Open(
//        &g_ospi_b_ctrl,
//        &g_ospi_b_cfg
//    );
//
//    if (ospi_err != FSP_SUCCESS)
//    {
//        rt_kprintf("OSPI open failed: %d\n", ospi_err);
//        set_led(1, 0, 0);
//        while (1) rt_thread_mdelay(1000);
//    }
//
//    // Initialize TFLite model
//    int model_status = motor_model_init();
//
//    if (model_status != 0)
//    {
//        rt_kprintf("Motor model init failed: %d\n", model_status);
//        set_led(1, 0, 0);
//        while (1) rt_thread_mdelay(1000);
//    }
//
//    rt_kprintf("Motor model initialized successfully\n");
//
//    set_led(1, 1, 1);  // WHITE: BOARD READY
//
//    while (1)
//    {
//            /* Wait for button press */
//            set_led(1, 1, 1);
//            do
//            {
//                R_IOPORT_PinRead(
//                    &g_ioport_ctrl,
//                    PIN_USER,
//                    &button_level
//                );
//
//                rt_thread_mdelay(10);
//
//            } while (button_level != BSP_IO_LEVEL_LOW);
//
//            /* Press debounce */
//            rt_thread_mdelay(30);
//
//            R_IOPORT_PinRead(
//                &g_ioport_ctrl,
//                PIN_USER,
//                &button_level
//            );
//
//            set_led(0, 0, 1);  // BLUE : COLLECTION
//            for (uint32_t chunk = 0; chunk < NUM_CHUNKS; chunk++)
//            {
//                g_done = 0;
//
//                /* Assert CS for this chunk */
//                R_IOPORT_PinWrite(
//                    &g_ioport_ctrl,
//                    PIN_CS,
//                    BSP_IO_LEVEL_LOW
//                );
//
//                /* Receive directly into the correct frame offset */
//                g_xfer_err = (uint32_t)
//                    g_spi1.p_api->writeRead(
//                        g_spi1.p_ctrl,
//                        tx_dummy,
//                        &rx_frames[chunk * CHUNK_SIZE],
//                        CHUNK_SIZE,
//                        SPI_BIT_WIDTH_8_BITS
//                    );
//
//                if (g_xfer_err != FSP_SUCCESS)
//                {
//                    R_IOPORT_PinWrite(
//                        &g_ioport_ctrl,
//                        PIN_CS,
//                        BSP_IO_LEVEL_HIGH
//                    );
//
//                    while (1)
//                        rt_thread_mdelay(1000);
//                }
//
//                /* Wait for completion */
//                uint32_t timeout = 100U;
//
//                while (!g_done && timeout > 0U)
//                {
//                    rt_thread_mdelay(1);
//                    timeout--;
//                }
//
//                if (!g_done)
//                {
//                    g_timeout++;
//
//                    R_IOPORT_PinWrite(
//                        &g_ioport_ctrl,
//                        PIN_CS,
//                        BSP_IO_LEVEL_HIGH
//                    );
//
//                    while (1)
//                        rt_thread_mdelay(1000);
//                }
//
//
//                R_IOPORT_PinWrite(
//                    &g_ioport_ctrl,
//                    PIN_CS,
//                    BSP_IO_LEVEL_HIGH
//                );
//
//                rt_thread_mdelay(2);
//            }
//            rt_thread_mdelay(300);  // Keep visible
//
//            set_led(0, 1, 0);  // GREEN: FRAME DONE
//            rt_thread_mdelay(500);
//
//            /* Complete frame received */
//            g_first  = rx_frames[0];
//            g_second = rx_frames[1];
//            g_third  = rx_frames[2];
//            g_last   = rx_frames[IMAGE_SIZE - 1U];
//
//            //g_frames++;
//            if (!validate_frame(rx_frames, IMAGE_SIZE))
//            {
//                set_led(1, 0, 0);  // RED
//                rt_thread_mdelay(4000);
//                return;
//            }
//            if (sizeof(rx_frames) == IMAGE_SIZE)
//            {
//                set_led(0, 1, 1);  // CYAN: MOTOR READY
//
//                int rs = motor_model_run(rx_frames, sizeof(rx_frames));
//
//                if (rs == 0)
//                {
//                    set_led(1, 1, 0);  // YELLOW: NO FAULT
//                    rt_kprintf("NO FAULT\n");
//                }
//                else if (rs == 1)
//                {
//                    set_led(1, 0, 0);  // RED: FAULT
//                    rt_kprintf("FAULT\n");
//                }
//                else
//                {
//                    rt_kprintf("Model run failed: %d\n", rs);
//                    set_led(1, 0, 0);
//                    while (1) rt_thread_mdelay(1000);
//                }
//            }
//            rt_thread_mdelay(1000);
//
//            /* Ready for next button press */
//            set_led(1, 1, 1);  // WHITE: READY
//
//    }
//
//}
