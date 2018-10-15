# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
from __future__ import division
import time
# Import the PCA9685 module.
import Adafruit_PCA9685
################################################################

import RPi.GPIO as GPIO
from time import sleep
import curses

#################### SET UP the STEERING WHEELS ################
# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)
#################################################################




class Motor:

    def __init__(self, pinForward1, pinBackward1,pinForward2, pinBackward2):
        """ Initialize the motor with its control pins and start pulse-width
             modulation """

        self.pinForward1 = pinForward1
        self.pinBackward1 = pinBackward1
        self.pinForward2 = pinForward2
        self.pinBackward2 = pinBackward2

        GPIO.setup(self.pinForward1, GPIO.OUT)
        GPIO.setup(self.pinBackward1, GPIO.OUT)
        GPIO.setup(self.pinForward2, GPIO.OUT)
        GPIO.setup(self.pinBackward2, GPIO.OUT)


        self.pwm_forward1 = GPIO.PWM(self.pinForward1, 100)
        self.pwm_backward1 = GPIO.PWM(self.pinBackward1, 100)
        self.pwm_forward2 = GPIO.PWM(self.pinForward2, 100)
        self.pwm_backward2 = GPIO.PWM(self.pinBackward2, 100)
        
        self.pwm_forward1.start(0)
        self.pwm_backward1.start(0)
        self.pwm_forward2.start(0)
        self.pwm_backward2.start(0)

    def forward(self, speed):
        """ pinForward is the forward Pin, so we change its duty
             cycle according to speed. """
        self.pwm_backward1.ChangeDutyCycle(0)
        self.pwm_forward1.ChangeDutyCycle(speed)
        self.pwm_backward2.ChangeDutyCycle(0)
        self.pwm_forward2.ChangeDutyCycle(speed)
        
    def backward(self, speed):
        """ pinBackward is the forward Pin, so we change its duty
             cycle according to speed. """
        self.pwm_backward1.ChangeDutyCycle(speed)
        self.pwm_forward1.ChangeDutyCycle(0)
        self.pwm_backward2.ChangeDutyCycle(speed)
        self.pwm_forward2.ChangeDutyCycle(0)

    def stop(self):
        """ Set the duty cycle of both control pins to zero to stop the motor. """
        self.pwm_forward1.ChangeDutyCycle(0)
        self.pwm_backward1.ChangeDutyCycle(0)
        self.pwm_forward2.ChangeDutyCycle(0)
        self.pwm_backward2.ChangeDutyCycle(0)

if __name__ == "__main__":
    #GPIO.cleanup()
    GPIO.setmode(GPIO.BOARD)
    motor = Motor(13, 15,11,12)
    
    
    ### set up the steering motor
    # Initialise the PCA9685 using the default address (0x40).
    pwm = Adafruit_PCA9685.PCA9685()
    # Configure min and max servo pulse lengths
    # Min = 150 , Max =600
    servo_min = 320  # Min pulse length out of 4096  (315)
# 21:00   315, 465, 390
# 22:02   310, 460, 385
# 22:03  320, 450, 385
# 22:06  325, 445, 385  
    servo_max = 440  # Max pulse length out of 4096  (465)
    servo_avg = 380 # to steer the wheel straight!!   (390)
    
    # Set frequency to 60hz, good for servos.
    pwm.set_pwm_freq(60)
    
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)

    try:
        while(True):
            char = screen.getch()
            if char == ord('q'):
                break
            elif char == curses.KEY_UP:
                motor.forward(35)
                #pwm.set_pwm(0, 0, 0)
                #sleep(0.1)
                pwm.set_pwm(0, 0, servo_avg)
                print("moving forward.")
            elif char == curses.KEY_DOWN:
                motor.backward(40)
                #pwm.set_pwm(0, 0, 0)
                #sleep(0.1)
                print("moving backward.")
                pwm.set_pwm(0, 0, servo_avg)
            elif char == curses.KEY_LEFT:
                motor.forward(40)
                #pwm.set_pwm(0, 0, 0)
                #sleep(0.1)
                pwm.set_pwm(0, 0, servo_min)
                print("moving left.")
            elif char == curses.KEY_RIGHT:
                motor.forward(35)
                #pwm.set_pwm(0, 0, 0)
                #sleep(0.1)
                pwm.set_pwm(0, 0, servo_max)
                print("moving right.")
            elif char == 10:
                motor.stop()
                print("take a rest.")
                pwm.set_pwm(0, 0, servo_avg)
                sleep(0.2)
                pwm.set_pwm(0, 0, 0)
                
    finally:
        curses.nocbreak()
        screen.keypad(False)
        curses.echo()
        curses.endwin()
        GPIO.cleanup()
        pwm.set_pwm(0, 0, servo_avg)
        sleep(0.2)
        pwm.set_pwm(0, 0, 0)
    
    # motor.forward(100)
    # sleep(2)
    # motor.stop()
    # sleep(2)

    # motor.forward_left(100)
    # sleep(2)
    # motor.stop()
    # sleep(2)

    # motor.forward_right(100)
    # sleep(2)
    # motor.stop()
    # sleep(2)
    
    # motor.left(25)
    # sleep(2)
    # motor.stop()
    # sleep(2)

    # motor.right(25)
    # sleep(2)
    # motor.stop()
    # sleep(2)

    
    # motor.backward(100)
    # sleep(2)
    # motor.stop()

    #GPIO.cleanup()
