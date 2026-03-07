// BioAmp EXG Pill - EEG Data Acquisition
// Optimized timing for stable 500 Hz sampling

#define SAMPLE_RATE 500
#define BAUD_RATE 115200
#define INPUT_PIN A0

// Hardware Configuration
#define ADC_RESOLUTION 1024.0          // 10-bit ADC
#define VREF 5.0                       // 5V reference (Arduino Uno)
#define VCENTER 2.5                    // Center voltage

// BioAmp EXG Pill Configuration
// Default gain - check if you soldered R6 for different gain
#define BIOAMP_GAIN 1000.0             

void setup() {
  Serial.begin(BAUD_RATE);
  analogReference(DEFAULT);
  
  // Stabilize ADC
  for(int i = 0; i < 10; i++) {
    analogRead(INPUT_PIN);
  }
}

void loop() {
  static unsigned long lastSampleTime = 0;
  unsigned long currentTime = micros();
  
  // Precise timing: sample every 2000 microseconds (500 Hz)
  if (currentTime - lastSampleTime >= 2000) {
    lastSampleTime = currentTime;
    
    // Read raw ADC value (0-1023)
    int raw_adc = analogRead(INPUT_PIN);
    
    // Convert ADC to voltage (0-5V)
    float voltage = (raw_adc / ADC_RESOLUTION) * VREF;
    
    // Convert to microvolts
    // Center the signal and scale by gain
    float microvolts = (voltage - VCENTER) * (1000000.0 / BIOAMP_GAIN);
    
    // Send with 2 decimal precision
    Serial.println(microvolts, 2);
  }
}