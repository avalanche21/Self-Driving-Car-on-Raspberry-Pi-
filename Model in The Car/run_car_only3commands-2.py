from __future__ import division
from time import sleep
import RPi.GPIO as GPIO
import cv2
import curses
import sys
import numpy as np
import Adafruit_PCA9685
# Import my own model
from forward_propagation import predict


#Motor function
class Motor:
    
    def __init__(self, pinForward1, pinBackward1,pinForward2, pinBackward2):
        """
        Initialize the motor with its control pins and start pulse-width
        modulation.
        The pinForward1, pinBackward1 control the left driving motor.
        The pinForward2, pinBackward2 control the right driving motor.
        """

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
        self.pwm_backward1.ChangeDutyCycle(0)
        self.pwm_forward1.ChangeDutyCycle(speed)
        self.pwm_backward2.ChangeDutyCycle(0)
        self.pwm_forward2.ChangeDutyCycle(speed)
        
    def backward(self, speed):
        self.pwm_backward1.ChangeDutyCycle(speed)
        self.pwm_forward1.ChangeDutyCycle(0)
        self.pwm_backward2.ChangeDutyCycle(speed)
        self.pwm_forward2.ChangeDutyCycle(0)

    def stop(self):
        self.pwm_forward1.ChangeDutyCycle(0)
        self.pwm_backward1.ChangeDutyCycle(0)
        self.pwm_forward2.ChangeDutyCycle(0)
        self.pwm_backward2.ChangeDutyCycle(0)


# Main function
if __name__ == '__main__':
    
    # setting up curses
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)
    
    # setting up GPIO pins
    GPIO.setmode(GPIO.BOARD)
    motor = Motor(13, 15, 11, 12)
    
    # setting up the steering motor.
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(60)
    servo_min = 320  # Min pulse length out of 4096
    servo_max = 440  # Max pulse length out of 4096
    servo_avg = 380  # steer the wheel straight!!!
    
    # load the parameters
    parameters = {}
    parameters['W1'] = np.load('W1.npy')
    parameters['W2'] = np.load('W2.npy')
    parameters['b1'] = np.load('b1.npy')
    parameters['b2'] = np.load('b2.npy')
    
    # setting up opencv
    cap = cv2.VideoCapture(0)
    cap.set(3,320) # set the width
    cap.set(4,240) # set the height
    cap.set(cv2.CAP_PROP_FPS, 20) # set frame per second (fps)
    assert(cap.get(cv2.CAP_PROP_FPS) == 20)
    
    try:
        # initialize self-driving
        raw_char = screen.getch()
        print("Press 'S' to start...")
        if raw_char == ord('s'):
            start_stop = True
        else:
            sys.exit(1)
        print("Now start self-driving...")
        screen.nodelay(True)

        # start self-driving
        while (start_stop):
            # read the image data from opencv
            _, img = cap.read()
                
            # cut, reshape, and scale the image data
            img = img[:, :, ::-1]
            img = np.reshape(img, (230400, 1))
            img = img.astype('float64')
            img /= 255
            
            # make the prediction
            predicted_key = np.argmax(predict(img, parameters))
            if predicted_key == 0:
                motor.forward(25)
                pwm.set_pwm(0, 0, servo_avg)
                print("forward")
            elif predicted_key == 1:
                motor.forward(25)
                pwm.set_pwm(0, 0, servo_min)
                print("moving left")
            elif predicted_key == 2:
                motor.forward(25)
                pwm.set_pwm(0, 0, servo_max)
                print("moving right")
            else:
                sys.exit(1)
        
            raw_char = screen.getch()
            if raw_char == ord('e'):
                motor.stop()
                pwm.set_pwm(0, 0, servo_avg)
                sleep(0.1)
                pwm.set_pwm(0, 0, 0)
                start_stop = False
    
    finally:
        # cleanup curses
        curses.nocbreak()
        screen.keypad(False)
        curses.echo(False)
        curses.endwin()
        
        # release OpenCV
        cap.release()
        cv2.destroyAllWindows()
        
        # cleanup GPIO
        GPIO.cleanup()
        print("The car is stopped.")
