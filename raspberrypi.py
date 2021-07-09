import paramiko, os
from time import sleep
import picamera
import RPi.GPIO as GPIO

LOCAL_DESKTOP = '/home/pi/Desktop/'
REMOTE_DESKTOP = '/home/jihyeon/Desktop/Garbage/data/target/target/'

# garbage - pin mapping
dic = {"glass": 5, "metal": 6, "plastic": 13, "cardboard": 4}

# camera setting
GPIO.setmode(GPIO.BCM);
GPIO.setwarnings(False)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # camera button

# led setting
GPIO.setup(13, GPIO.OUT)  # (1)plastic-red led
GPIO.setup(6, GPIO.OUT)  # (2)metal-blue led
GPIO.setup(5, GPIO.OUT)  # (3)glass-red led
GPIO.setup(4, GPIO.OUT)  # (4)cardboard-blue led

# buzzer setting
GPIO.setup(22, GPIO.OUT)
p = GPIO.PWM(22, 100)
p.start(10)
p.ChangeDutyCycle(0)


def put_file(src, dst):
    transport = paramiko.Transport('114.70.193.160', 22)
    transport.connect(username='jihyeon', password='wlgus5273!')
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(src, dst)
    sftp.close()
    transport.close()


def getResult():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        ssh.connect('114.70.193.160', username='jihyeon', password='wlgus5273!')

        put_file(os.path.join(LOCAL_DESKTOP, 'image.jpg'),
                 os.path.join(REMOTE_DESKTOP, 'target.jpg'))

        stdin, stdout, stderr = ssh.exec_command('~/anaconda3/envs/my_env/bin/python ~/Desktop/Garbage/result.py')

        errs = stderr.readlines()
        for err in errs:
            print(err)
        output = stdout.readlines()[0]

        ssh.close()

        return output

    except Exception as err:
        print(err)


def ViewResult(result):
    p.ChangeDutyCycle(10)  # result alarm
    sleep(0.1)
    p.ChangeDutyCycle(0)
    led_pin = dic[result.split('\n')[0]]
    GPIO.output(led_pin, 1)
    sleep(3)
    GPIO.output(led_pin, 0)


try:
    while True:  # camera capture
        with picamera.PiCamera() as camera:
            camera.resolution = (512, 384)
            camera.start_preview()
            while GPIO.input(17) == 0:  # button not push
                sleep(0.1)
            sleep(1)
            camera.capture('image.jpg')
            p.ChangeDutyCycle(10)  # capture alarm
            sleep(0.1)
            p.ChangeDutyCycle(0)
            result = getResult()
            print(result)
            camera.stop_preview()

        # from os import system
        # system("python3.7 /home/pi/webapps/Adafruit_Python_SSD1306/examples/shapes.py --result='"+result+"'")

        ViewResult(result)

except KeyboardInterrupt:
    pass

GPIO.cleanup()
p.stop()