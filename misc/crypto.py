"""Crypto
    Handles key encryption using AES to ensure keys are protected
    Supports encryption of keys for multiple exchanges
"""

import getpass
import hashlib
import json
import os.path
import re

from Crypto.Cipher import AES

base_file = 'data/'

def save_to_file(filename, iv, encrypted):
    f = open(base_file + filename, 'wb')
    f.write(iv + encrypted)
    f.close()
    print("File saved")

def add_exchange(exchange_name=None, filename=None, pw=''):
    # If no exchange provided, get the name
    if not exchange_name:
        exchange_name = input("Exchange name: ").upper()
        
    # Input details
    public = input("API Public key: ")
    key = getpass.getpass("API Secret: ")
    
    data = {'exchange':exchange_name,
             'public':public,
             'private':key}

    if pw == '':
        pw = '1'
        pw2 = '2'
        while pw != pw2:
            pw = getpass.getpass("Password to encrypt API info: ")
            pw2 = getpass.getpass("Repeat password: ")
            if pw != pw2:
                print("Passwords do not match")
        
    new_pw = hashlib.sha256(pw.encode()).digest()
    
    data = json.dumps(data)
    iv = hashlib.sha256(public.encode()).digest()
    iv = iv[:AES.block_size]

    obj = AES.new(new_pw, AES.MODE_CFB, iv)
    encrypted = obj.encrypt(data.encode("utf8"))
    if filename:
        save_to_file(filename, iv, encrypted)
    return iv + encrypted

def get_api_info(data=None, pw='', filename='key.dat'):
    
    if not data and not os.path.isfile(base_file + filename):
        data = add_exchange(filename=filename)

    if not data:
        with open(base_file + filename, 'rb') as file:
            data = file.read()
    
    if pw == '':
        pw = getpass.getpass()
    split = AES.block_size
    iv = data[:split]
    encrypted = data[split:]
    
    pw = hashlib.sha256(pw.encode()).digest()
    obj = AES.new(pw, AES.MODE_CFB, iv)
    try:
        decrypted = obj.decrypt(encrypted)
        decrypted = decrypted.decode("utf8")
        data = json.loads(decrypted)
    except:
        print('Error: decrypt failed')
        return ''
 
    return data
    
if __name__ == '__main__':
    add_exchange(filename='key.dat')
