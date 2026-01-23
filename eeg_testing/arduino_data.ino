

#define SAMPLE_RATE 500
#define BAUD_RATE 115200
#define INPUT_PIN A0

void setup() {
  Serial.begin(BAUD_RATE);
  analogReference(DEFAULT);
}

void loop() {
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;
  
  static long timer = 0;
  timer -= interval;
  
  if(timer < 0) {
    timer += 1000000 / SAMPLE_RATE;
    
    
    int raw_adc = analogRead(INPUT_PIN);
    
    
    Serial.println(raw_adc);
  }
}
