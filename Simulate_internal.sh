#!/bin/bash


USERNAMES=("user1" "user2" "admin" "testuser")

INVALID_PASSWORD="wrongpassword123"

ATTEMPTS_PER_USER=5

echo "Simulating failed login attempts..."
for USER in "${USERNAMES[@]}"; do
    echo "Trying to switch to account: $USER"
    for i in $(seq 1 $ATTEMPTS_PER_USER); do
        echo "Attempt $i for user $USER"
        echo "$INVALID_PASSWORD" | su - $USER > /dev/null 2>&1
        
        if [ $? -ne 0 ]; then
            echo "Failed login attempt $i for user: $USER"
        fi
        sleep 1  
    done
    echo "Finished attempts for user: $USER"
done

echo "Simulation complete."
