#! /bin/sh

ip=`/sbin/ifconfig  | grep "Bcast" | sed 's/.*inet addr:[0-9]*\.[0-9]*\.[0-9]*\.\([0-9]*\).*/\1/g'`

if [ -z "$ip" ]; then
	ip="000"
fi

autossh -f -N -R 0.0.0.0:22$ip:127.0.0.1:22 maxim@9lab.ru
autossh -f -N -R 0.0.0.0:8$ip:127.0.0.1:30211 maxim@9lab.ru
