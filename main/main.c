#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"

#include "driver/gptimer.h"
#include "esp_adc/adc_oneshot.h"

#define TAG "EEG"

// -------- EEG CONFIG --------
#define EEG_SAMPLE_RATE   250               // Hz
#define EEG_BUFFER_SIZE   EEG_SAMPLE_RATE   // 1 second buffer

#define EEG_ADC_UNIT      ADC_UNIT_1
#define EEG_ADC_CHANNEL   ADC_CHANNEL_4     // GPIO32

static adc_oneshot_unit_handle_t adc_handle;
static gptimer_handle_t timer_handle;
static QueueHandle_t timer_queue;

static uint16_t eeg_buffer[EEG_BUFFER_SIZE];
static int buffer_index = 0;

// -------- TIMER CALLBACK --------
static bool IRAM_ATTR timer_callback(gptimer_handle_t timer,
                                     const gptimer_alarm_event_data_t *edata,
                                     void *user_data)
{
    int evt = 1;
    xQueueSendFromISR(timer_queue, &evt, NULL);
    return true;
}

// -------- EEG TASK --------
void eeg_task(void *arg)
{
    int evt;
    int raw;

    while (1)
    {
        if (xQueueReceive(timer_queue, &evt, portMAX_DELAY))
        {
            adc_oneshot_read(adc_handle, EEG_ADC_CHANNEL, &raw);
            eeg_buffer[buffer_index++] = raw;

            if (buffer_index >= EEG_BUFFER_SIZE)
            {
                for (int i = 0; i < EEG_BUFFER_SIZE; i++)
                {
                    printf("%d,", eeg_buffer[i]);
                }
                printf("\n");
                buffer_index = 0;
            }
        }
    }
}

// -------- MAIN --------
void app_main(void)
{
    ESP_LOGI(TAG, "EEG Recorder (ESP-IDF v5.x)");

    // ADC CONFIG
    adc_oneshot_unit_init_cfg_t adc_cfg = {
        .unit_id = EEG_ADC_UNIT,
    };
    adc_oneshot_new_unit(&adc_cfg, &adc_handle);

    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_12,
    };
    adc_oneshot_config_channel(adc_handle, EEG_ADC_CHANNEL, &chan_cfg);

    // Queue
    timer_queue = xQueueCreate(10, sizeof(int));

    // GPTIMER CONFIG
    gptimer_config_t timer_cfg = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    gptimer_new_timer(&timer_cfg, &timer_handle);

    gptimer_event_callbacks_t cbs = {
        .on_alarm = timer_callback,
    };
    gptimer_register_event_callbacks(timer_handle, &cbs, NULL);

    gptimer_alarm_config_t alarm_cfg = {
        .reload_count = 0,
        .alarm_count = 1000000 / EEG_SAMPLE_RATE,
        .flags.auto_reload_on_alarm = true,
    };
    gptimer_set_alarm_action(timer_handle, &alarm_cfg);

    gptimer_enable(timer_handle);
    gptimer_start(timer_handle);

    xTaskCreatePinnedToCore(
        eeg_task,
        "eeg_task",
        4096,
        NULL,
        5,
        NULL,
        1
    );
}
