TARGET_HOST="192.168.1.100"  
PORT_START=1                  
PORT_END=100            
DELAY_BETWEEN_PROBES=0.2  

echo "Simulating port-scan against $TARGET_HOST from $(hostname)…"
echo "Ports: $PORT_START–$PORT_END  |  Delay: ${DELAY_BETWEEN_PROBES}s"

for PORT in $(seq "$PORT_START" "$PORT_END"); do
    echo "Probing TCP port $PORT"
    nc -z -w1 "$TARGET_HOST" "$PORT" &>/dev/null
    sleep "$DELAY_BETWEEN_PROBES"
done

echo "Simulation complete."
