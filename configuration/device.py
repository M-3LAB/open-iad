import socket
import fcntl
import struct
from configuration.registration import server_data


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', bytes(ifname[:15], 'utf-8')))

    return socket.inet_ntoa(info[20:24])

def assign_service(moda='eno1'):
    # moda: eno1, lo
    ip = get_ip_address(moda)
    root_path = server_data[ip]

    return ip, root_path

