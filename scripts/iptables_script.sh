sudo iptables -A INPUT -p tcp --dport 3001 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --sport 3001 -j ACCEPT
