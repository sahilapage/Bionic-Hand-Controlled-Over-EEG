// BioAmp EXG Pill -> Arduino Uno R3, single-channel EEG at 500 Hz.
//
// Prints one microvolt reading per line over serial. `sohand.eeg.bridge`
// republishes that as the LSL stream `BioAmp_EXG`.
//
// GAIN must match the board. The EXG Pill ships at 1000x; if you have
// re-soldered the gain resistor, change BIOAMP_GAIN or every downstream
// microvolt figure is wrong by that factor. Band *ratios* are unaffected,
// which is why the classifier still works with a mis-set gain -- the absolute
// numbers are what stop meaning anything.

#define SAMPLE_RATE     500       // Hz
#define BAUD_RATE       115200    // 8 bytes/sample at 500 Hz needs >= 57600
#define INPUT_PIN       A0

#define ADC_RESOLUTION  1024.0    // 10-bit ADC
#define VREF            5.0       // Arduino Uno reference, volts
#define VCENTER         2.5       // the EXG Pill biases the signal to mid-rail
#define BIOAMP_GAIN     1000.0

const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE;

void setup() {
  Serial.begin(BAUD_RATE);
  analogReference(DEFAULT);

  // The first conversions after switching the mux settle on the sample-and-hold
  // capacitor rather than the input; discard them.
  for (int i = 0; i < 10; i++) {
    analogRead(INPUT_PIN);
  }
}

void loop() {
  static unsigned long lastSampleTime = 0;
  unsigned long now = micros();

  // Comparing the elapsed difference (not `now >= next`) keeps the timing
  // correct across the ~70 minute micros() rollover.
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = now;

    int raw = analogRead(INPUT_PIN);
    float volts = (raw / ADC_RESOLUTION) * VREF;
    float microvolts = (volts - VCENTER) * (1000000.0 / BIOAMP_GAIN);

    Serial.println(microvolts, 2);
  }
}
