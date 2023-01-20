import socket
import fcntl
import struct

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', bytes(ifname[:15], 'utf-8')))

    return socket.inet_ntoa(info[20:24])

def assign_service(guoyang):
    if guoyang:
        ip = get_ip_address('lo')
    else:
        ip = get_ip_address('eno1')

    root_path = None

    if ip == '172.18.36.46':
        root_path = '/disk4/xgy' 
    if ip == '127.0.0.1':
        root_path = '/home/robot/data'
    if ip == '172.18.34.25':
        root_path = '/home/zhengf_lab/cse30010351/m3lab/data'
    if ip == '172.18.36.107':
        root_path = '/ssd-sata1/wjb/data/open-ad'
    if ip == '172.18.36.108':
        root_path = '/ssd2/m3lab/data/open-ad'

    return ip, root_path

