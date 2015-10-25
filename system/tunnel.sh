#! /bin/sh

ip=`/sbin/ifconfig  | grep "Bcast" | sed 's/.*inet addr:[0-9]*\.[0-9]*\.[0-9]*\.\([0-9]*\).*/\1/g'`

if [ -z "$ip" ]; then
	ip="000"
fi

autossh -f -N -R 22$ip:127.0.0.1:22 maxim@9lab.ru
