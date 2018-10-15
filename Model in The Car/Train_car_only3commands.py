from __future__ import division
from time import sleep
import RPi.GPIO as GPIO
import cv2
import curses
import numpy as np
import Adafruit_PCA9685
import sys

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
    
    # setting up the speed of the car
    # car_speed = int(sys.argv[1])
    # print("Our car runs at speed", car_speed)
    
    # Setting up the resolution parameter.
    res_setting = "Low"
    assert(res_setting == "High" or res_setting == "Low")
    
    # setting up the driving motor.
    GPIO.setmode(GPIO.BOARD)
    motor = Motor(13, 15, 11, 12)
    
    # setting up the steering motor.
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(60)
    servo_min = 320  # Min pulse length out of 4096
    servo_max = 440  # Max pulse length out of 4096
    servo_avg = 380  # steer the wheel straight!!!
    
    # setting up curses.
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)
    
    # setting up OpenCV.
    cap = cv2.VideoCapture(0)
    if res_setting == "High":
        cap.set(3, 600)  #set the width
        cap.set(4, 450)  #set the height
    elif res_setting == "Low":
        cap.set(3, 320)  #set the width
        cap.set(4, 240)  #set the height
    cap.set(cv2.CAP_PROP_FPS, 20)
    assert(cap.get(cv2.CAP_PROP_FPS) == 20)
    
    # setting up two lists for storing data.
    data_frames = []
    data_commands = []
    
    try:
        # initialize self-driving.
        raw_char = screen.getch()
        print("Press 'S' to start...")
        if raw_char == ord('s'):
            start_stop = True
        else:
            sys.exit(1)
        print("Now collecting data...")
        sleep(0.2)
        screen.nodelay(True)
        char = 0
        
        # collecting training data.
        while (start_stop):
            
            _, img = cap.read()
            
            raw_char = screen.getch()
            if raw_char == curses.KEY_UP:
                char = 0
                motor.forward(35)
                pwm.set_pwm(0, 0, servo_avg)
            elif raw_char == curses.KEY_LEFT:
                char = 1
                motor.forward(35)
                pwm.set_pwm(0, 0, servo_min)
            elif raw_char == curses.KEY_RIGHT:
                char = 2
                motor.forward(35)
                pwm.set_pwm(0, 0, servo_max)
            elif raw_char == ord('e'):
                start_stop = False
                motor.stop()
                pwm.set_pwm(0, 0, servo_avg)
                sleep(0.2)
                pwm.set_pwm(0, 0, 0)
            
            # put all data into lists.
            data_commands.append(char)
            if res_setting == "High":
                img = img[149:-1, :, :]
            elif res_setting == "Low":
                img = img[:, :, ::-1]
                #img = img[119:-1, :, ::-1]
            data_frames.append(img)
    
    finally:
        
        # Clean up curses
        curses.nocbreak()
        screen.keypad(False)
        curses.echo(False)
        curses.endwin()
    
        # release OpenCV
        cap.release()
        cv2.destroyAllWindows()
        
        # clean up GPIO
        GPIO.cleanup()
        
        # data_commands = np.array(data_commands)    
        # print(len(data_frames))
        # print(data_commands.shape)
        # print(len(data_frames))
        # print(data_frames[50].shape)
        # print(data_commands)
        # print(data_frames[50])
        # print(data_frames[10].shape)
        # print(data_frames[10][1])
        # x_train = np.concatenate(data_frames)
        
        # compiling the data.
        x_train = np.array(data_frames)
        y_train = np.array(data_commands)
        assert(x_train.shape[0] == y_train.shape[0])
        print("Our X_training data has shape:", x_train.shape)
        print("Our Y_training data has shape:", x_train.shape)

        # saving the data.
        if res_setting == "High":
            np.save('x_train_high_res.npy', x_train)   
            np.save('y_train_high_res.npy', y_train)
        elif res_setting == "Low":
            np.save('x_train_low_res.npy', x_train)   
            np.save('y_train_low_res.npy', y_train)

