#define SAMPLE_RATE 256
#define BAUD_RATE 115200
#define INPUT_PIN A0
#define THRESHOLD 1000

void setup() {
  Serial.begin(BAUD_RATE);
}

void loop() {
  static unsigned long lastMicros = 0;
  unsigned long now = micros();

  if (now - lastMicros >= 1000000UL / SAMPLE_RATE) {
    lastMicros = now;

    float raw = analogRead(INPUT_PIN);
    float filtered = EEGFilter(raw);

    // Use absolute value or power
    float value1 = filtered * filtered;   // signal power
    int flag = (value1 > THRESHOLD) ? 1 : 0;

    // Send to Serial Plotter
    Serial.print(value1);
    Serial.print(",");
    Serial.println(flag);
  }
}

// -------------------------------------------------
float EEGFilter(float input) {
  float output = input;

  {
    static float z1, z2;
    float x = output - -0.95391350*z1 - 0.25311356*z2;
    output = 0.00735282*x + 0.01470564*z1 + 0.00735282*z2;
    z2 = z1; z1 = x;
  }
  {
    static float z1, z2;
    float x = output - -1.20596630*z1 - 0.60558332*z2;
    output = x + 2*z1 + z2;
    z2 = z1; z1 = x;
  }
  {
    static float z1, z2;
    float x = output - -1.97690645*z1 - 0.97706395*z2;
    output = x - 2*z1 + z2;
    z2 = z1; z1 = x;
  }
  {
    static float z1, z2;
    float x = output - -1.99071687*z1 - 0.99086813*z2;
    output = x - 2*z1 + z2;
    z2 = z1; z1 = x;
  }
  return output;
}
