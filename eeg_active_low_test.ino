#define SAMPLE_RATE 256
#define BAUD_RATE 115200
#define INPUT_PIN A0

#define WINDOW_SIZE 256   // 1 second window

// Power accumulators
float alphaSum = 0;
float betaSum  = 0;
int sampleCount = 0;

void setup() {
  Serial.begin(BAUD_RATE);
}

void loop() {
  static unsigned long lastMicros = 0;
  unsigned long now = micros();

  if (now - lastMicros >= 1000000UL / SAMPLE_RATE) {
    lastMicros = now;

    // 1) Read EEG signal
    float raw = analogRead(INPUT_PIN);

    // 2) Remove DC offset (VERY IMPORTANT)
    raw = removeDC(raw);

    // 3) Filter into EEG bands
    float alpha = AlphaFilter(raw);
    float beta  = BetaFilter(raw);

    // 4) Accumulate signal power
    alphaSum += alpha * alpha;
    betaSum  += beta  * beta;
    sampleCount++;

    // 5) Every 1 second, classify mental state
    if (sampleCount >= WINDOW_SIZE) {

      // Log power (EEG standard)
      float alphaPower = log(alphaSum / WINDOW_SIZE + 1);
      float betaPower  = log(betaSum  / WINDOW_SIZE + 1);

      float diff = betaPower - alphaPower;

      /*
        state:
        0 = RELAXED
        1 = EXCITED
        2 = NEUTRAL
      */
      int state;
      if (diff > 0.15)       state = 1; // EXCITED
      else if (diff < -0.15) state = 0; // RELAXED
      else                   state = 2; // NEUTRAL

      // Output for Serial Plotter
      Serial.print(alphaPower);
      Serial.print(",");
      Serial.print(betaPower);
      Serial.print(",");
      Serial.println(state);

      // Reset window
      alphaSum = 0;
      betaSum  = 0;
      sampleCount = 0;
    }
  }
}

/* =====================================================
   DC OFFSET REMOVAL (High-pass ~0.5 Hz)
   ===================================================== */
float removeDC(float x) {
  static float mean = 0;
  mean = 0.999 * mean + 0.001 * x;
  return x - mean;
}

/* =====================================================
   ALPHA BAND FILTER (8–13 Hz, Fs = 256 Hz)
   ===================================================== */
float AlphaFilter(float input) {
  static float z1 = 0, z2 = 0;
  float output;

  float x = input - (-1.5610181f * z1) - (0.6413515f * z2);
  output = 0.0200834f * x + 0.0401668f * z1 + 0.0200834f * z2;

  z2 = z1;
  z1 = x;

  return output;
}

/* =====================================================
   BETA BAND FILTER (13–30 Hz, Fs = 256 Hz)
   ===================================================== */
float BetaFilter(float input) {
  static float z1 = 0, z2 = 0;
  float output;

  float x = input - (-0.3695274f * z1) - (0.1958157f * z2);
  output = 0.2065721f * x - 0.4131442f * z1 + 0.2065721f * z2;

  z2 = z1;
  z1 = x;

  return output;
}
