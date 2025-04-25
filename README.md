## Правила auditd
-a always,exit -F arch=b64 -S execve  
-w /var/log/auth.log -p rwxa -k auth_logs
-a always,exit -F arch=b64 -S execve,execveat \ -F exe=/usr/bin/nmap        -k recon_exec
-a always,exit -F arch=b64 -S execve,execveat \ -F exe=/usr/bin/masscan     -k recon_exec
-a always,exit -F arch=b64 -S execve,execveat \ -F exe=/usr/bin/nping       -k recon_exec
-a always,exit -F arch=b64 -S execve,execveat \ -F exe=/bin/nc              -k recon_exec
-a always,exit -F arch=b64 -S execve,execveat \ -F exe=/usr/bin/ncat        -k recon_exec
-a always,exit -F arch=b64 -S execve,execveat \ -F exe=/usr/bin/hping3      -k recon_exec


## Команды поиска связанных событий в auditd
ps -o pid,ppid,cmd -p <PID>  
cat /proc/<PID>/status | grep PPid  
pstree -p <PID>  
ausearch -m USER_AUTH --success no | awk -F 'addr=' '{print $2}' | awk '{print $1}' | sort | uniq -c | sort -nr
