import Jetson.GPIO as GPIO
import time

#modo de enumeracion de los pines 
GPIO.setmode(GPIO.BOARD)
#numero del pin
outPin = 12
#configuracion del pin
GPIO.setup(outPin, GPIO.OUT)
#primer valor del pin
value = GPIO.HIGH

#ciclo infinito
while True:
    #pausar por 1 segundo
    time.sleep(1)
    #cambiar valor con XOR
    value ^= GPIO.HIGH
    #asignar nuevo valor
    GPIO.output(outPin, value)
    #imrpimir valor
    print(value)