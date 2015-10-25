#! /bin/sh

LDIR="/home/pi/Level"

cd $LDIR

git pull -r

crontab ./system/crontab.cfg