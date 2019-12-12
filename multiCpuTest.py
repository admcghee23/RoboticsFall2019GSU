#!/usr/bin/env python

'''
    multiCpuTest.py - Application to demonstrate the use of a processor's multiple CPUs.
    
    This capability is very handy when a robot needs more processor power, and has processing elements
    that can be cleaved off to another CPU, and work in parallel with the main application.
    
    https://docs.python.org/2/library/multiprocessing.html  describes the many ways the application parts
		can communicate, beyond this simple example

    Copyright (C) 2017 Rolly Noel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
'''

from multiprocessing import cpu_count, Process, Value, Array
import time

secondCpuToRun = Value('i', False)  #Shared memory designation for an integer
timeLastSaw = Value('d', 0.)         #Shared memory designation for a decimal number

def watchTheTime(timeLastSaw): 	#THIS FUNCTION RUNS ON ITS OWN CPU
    print()
    while secondCpuToRun.value:
        now = time
        timeLastSaw.value = time.time()     #
        print('2nd CPU reporting the time: %d' % timeLastSaw.value)
        time.sleep(5)           #Sleep for 5 seconds
    print('Second CPU task shutting down')

if __name__ == '__main__':
    print("System has %d CPUs" % cpu_count())
    secondCpuToRun.value = True
    proc = Process(target=watchTheTime, args=(timeLastSaw,)) 	#Consume another cpu - TRAILING COMMA NEEDED
    proc.daemon = True
    proc.start()
    x = raw_input("Hit Enter to shut 2nd CPU down")     #This CPU is blocked till user hits Enter
    secondCpuToRun.value = False                #Tell second CPU process to shut down
    time.sleep(1)                               #Give it a chance
    print("Last time 2nd CPU captured was %d" % timeLastSaw.value)  #Show work 2nd CPU did

