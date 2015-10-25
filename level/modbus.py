#!/usr/bin/env python
#---------------------------------------------------------------------------# 
# the various server implementations
#---------------------------------------------------------------------------# 
from pymodbus.server.async import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock #, ModbusDeviceIdentification
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.device import ModbusDeviceIdentification
from twisted.internet.task import LoopingCall
import subprocess, re

import RPi.GPIO as GPIO

GPIO.setwarnings(False)

GPIO.cleanup()

#---------------------------------------------------------------------------# 
# configure the service logging
#---------------------------------------------------------------------------# 
import logging
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ledState = None
ledProc = None

def led(state):
    global ledState, ledProc
    if state == ledState:
        return
    
    ledState = state
    if ledProc:
        ledProc.stop()
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)
    freq = 5 if state else 0.5
    fillness = 50 if state else 95
    pin = 11
    ledProc = GPIO.PWM(pin, freq)
    ledProc.start(fillness)

def updating_writer(a):
    
    
    context  = a[0]
    register = 4
    slave_id = 0x00
    ok = 0
    error = 1
    
    try:
        line = subprocess.check_output(['tail', '-1', '/tmp/level.val'])
        level = min(max(int(float(line)*100), 0), 10000)
    except:
        led(error)
    else:
        context[slave_id].setValues(4,0x00,[level])
        led(ok)
        
    try:
        line = subprocess.check_output(['tail', '-1', '/home/pi/logs/dht.log'])
        matches = re.findall("Temp=(.*)\*  Humidity=(.*)%", line)
        temp =  min(max(int(float(matches[0][0])*10), 0), 10000)
        humidity =  min(max(int(float(matches[0][1])*10), 0), 10000)
    except:
        pass
    else:
        context[slave_id].setValues(4,0x01,[temp])
        context[slave_id].setValues(4,0x02,[humidity])

#---------------------------------------------------------------------------# 
# initialize your data store
#---------------------------------------------------------------------------# 
store = ModbusSlaveContext(
    di = ModbusSequentialDataBlock(0, [0]*100),
    co = ModbusSequentialDataBlock(0, [0]*100),
    hr = ModbusSequentialDataBlock(0, [0]*100),
    ir = ModbusSequentialDataBlock(0, [0]*100))

context = ModbusServerContext(slaves=store, single=True)

identity = ModbusDeviceIdentification()
identity.VendorName  = 'Pymodbus'
identity.ProductCode = 'PM'
identity.VendorUrl   = 'http://github.com/bashwork/pymodbus/'
identity.ProductName = 'Pymodbus Server'
identity.ModelName   = 'Pymodbus Server'
identity.MajorMinorRevision = '1.0'

loop = LoopingCall(f=updating_writer, a=(context,))
loop.start(1, now=False)
StartTcpServer(context, identity=identity)
