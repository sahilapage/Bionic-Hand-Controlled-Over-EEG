#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_timer.h"

// Configuration
#define SAMPLE_RATE 75
#define INPUT_PIN ADC_CHANNEL_6  // GPIO34 for ESP32
#define DATA_LENGTH 10

// SENSITIVITY ADJUSTMENT - Change this value to tune blink detection
// Lower value = MORE SENSITIVE (detects smaller blinks)
// Higher value = LESS SENSITIVE (only detects strong blinks)
// Recommended range: 0.4 (very sensitive) to 2.0 (less sensitive)
// Default was 1.2, now set to 0.8 for better detection
#define SENSITIVITY_THRESHOLD 0.5

// ADC Configuration
#define ADC_ATTEN ADC_ATTEN_DB_12
#define ADC_WIDTH ADC_BITWIDTH_12

// Global variables
static int data_index = 0;
static bool peak = false;
static adc_oneshot_unit_handle_t adc1_handle;
static adc_cali_handle_t adc1_cali_handle = NULL;

// Function prototypes
float EOGFilter(float input);
bool Getpeak(float new_sample);

void app_main(void)
{
    // Configure ADC
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = ADC_UNIT_1,
    };
    adc_oneshot_new_unit(&init_config, &adc1_handle);
    
    // Configure ADC channel
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_WIDTH,
        .atten = ADC_ATTEN,
    };
    adc_oneshot_config_channel(adc1_handle, INPUT_PIN, &config);
    
    // ADC Calibration Init (Line Fitting for ESP32)
    adc_cali_line_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT_1,
        .atten = ADC_ATTEN,
        .bitwidth = ADC_WIDTH,
    };
    adc_cali_create_scheme_line_fitting(&cali_config, &adc1_cali_handle);
    
    printf("EOG Signal Monitor Started\n");
    printf("signal,peak\n");  // CSV header for plotting
    
    uint64_t past = esp_timer_get_time();
    int64_t timer = 0;
    const int64_t sample_interval = 1000000 / SAMPLE_RATE;
    
    while(1) {
        // Calculate elapsed time
        uint64_t present = esp_timer_get_time();
        int64_t interval = present - past;
        past = present;
        
        // Run timer
        timer -= interval;
        
        // Sample at defined rate
        if(timer < 0) {
            timer += sample_interval;
            
            // Read ADC value
            int adc_raw = 0;
            adc_oneshot_read(adc1_handle, INPUT_PIN, &adc_raw);
            
            // Normalize input data (-1 to 1)
            float sensor_value = (float)adc_raw;
            float signal = EOGFilter(sensor_value) / 512.0;
            
            // Get peak
            peak = Getpeak(signal);
            
            // Print for Serial Plotter/Monitor
            printf("%.4f,%d\n", signal, peak ? 1 : 0);
        }
        
        // Small delay to prevent watchdog timeout
        vTaskDelay(1);
    }
}

bool Getpeak(float new_sample) {
    // Buffers for data, mean, and standard deviation
    static float data_buffer[DATA_LENGTH] = {0};
    static float mean_buffer[DATA_LENGTH] = {0};
    static float standard_deviation_buffer[DATA_LENGTH] = {0};
    static float mean = 0.0;
    float standard_deviation = 0.0;
    
    // Store old value for mean calculation
    float old_value = data_buffer[data_index];
    
    // SENSITIVITY ADJUSTMENT: Lower value = more sensitive
    // Default: DATA_LENGTH * 1.2 = 12.0
    // More sensitive: Try 8.0, 6.0, or even 4.0
    float threshold = DATA_LENGTH * SENSITIVITY_THRESHOLD;
    
    // Check for peak
    if (new_sample - mean_buffer[data_index] > threshold * standard_deviation_buffer[data_index]) {
        peak = true;
    } else {
        peak = false;
    }
    
    // Update mean dynamically (sliding window mean)
    mean = mean + (new_sample - old_value) / DATA_LENGTH;
    
    // Update data buffer
    data_buffer[data_index] = new_sample;
    
    // Calculate standard deviation
    for (int i = 0; i < DATA_LENGTH; ++i) {
        standard_deviation += pow(data_buffer[i] - mean, 2);
    }
    
    // Update mean buffer
    mean_buffer[data_index] = mean;
    
    // Update standard deviation buffer
    standard_deviation_buffer[data_index] = sqrt(standard_deviation / DATA_LENGTH);
    
    // Update data_index
    data_index = (data_index + 1) % DATA_LENGTH;
    
    // Return peak
    return peak;
}

// Band-Pass Butterworth IIR digital filter
// Sampling rate: 75.0 Hz, frequency: [0.5, 19.5] Hz.
float EOGFilter(float input)
{
    float output = input;
    {
        static float z1 = 0, z2 = 0;
        float x = output - 0.02977423*z1 - 0.04296318*z2;
        output = 0.09797471*x + 0.19594942*z1 + 0.09797471*z2;
        z2 = z1;
        z1 = x;
    }
    {
        static float z1 = 0, z2 = 0;
        float x = output - 0.08383952*z1 - 0.46067709*z2;
        output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
        z2 = z1;
        z1 = x;
    }
    {
        static float z1 = 0, z2 = 0;
        float x = output - -1.92167271*z1 - 0.92347975*z2;
        output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
        z2 = z1;
        z1 = x;
    }
    {
        static float z1 = 0, z2 = 0;
        float x = output - -1.96758891*z1 - 0.96933514*z2;
        output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
        z2 = z1;
        z1 = x;
    }
    return output;
}