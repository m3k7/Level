#! /bin/sh

while true
do
	sleep 5
	PYTHONPATH=$PYTHONPATH:/home/pi/PyContrib:/home/pi/Level python3 /home/pi/Level/level/service.py --config /home/pi/Level/configuration/level-prod.cfg
done