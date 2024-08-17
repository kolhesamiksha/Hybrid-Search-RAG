import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import ast

def decrypt_pass(encrypted_key):
    encrypted_key = ast.literal_eval(encrypted_key)
    iv = encrypted_key[:AES.block_size]  # Extract the IV
    cipher_text = encrypted_key[AES.block_size:]  # Extract the ciphertext
    cipher = AES.new(iv, AES.MODE_CBC, iv)  # Recreate the cipher with the extracted IV
    padded_plain_text = cipher.decrypt(cipher_text)
    plain_text = unpad(padded_plain_text, AES.block_size)
    return plain_text.decode('utf-8')