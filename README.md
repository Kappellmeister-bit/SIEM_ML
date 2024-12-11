## Правила auditd
sudo auditctl -a always,exit -F arch=b64 -S execve  
sudo auditctl -w /var/log/auth.log -p rwxa -k auth_logs


## Команды поиска связанных событий в auditd
ps -o pid,ppid,cmd -p <PID>
cat /proc/<PID>/status | grep PPid
pstree -p <PID>
ausearch -m USER_AUTH --success no | awk -F 'addr=' '{print $2}' | awk '{print $1}' | sort | uniq -c | sort -nr
