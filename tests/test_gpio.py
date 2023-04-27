import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
outPin = 12

GPIO.setup(outPin, GPIO.OUT, initial=GPIO.HIGH)
value = GPIO.HIGH
while True:
    time.sleep(1)
    value ^= GPIO.HIGH
    GPIO.output(outPin,value)
    print(value)