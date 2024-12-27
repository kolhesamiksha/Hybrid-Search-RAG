import os
import base64
import traceback
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from hybrid_rag.src.utils.logutils import Logger

logger = Logger().get_logger()

class AESDecryptor:
    def __init__(self):
        """
        Initializes the AESDecryptor with the base64 encoded key.

        """

    def __decrypt_pass(self, base64_encoded_key) -> str:
        """
        Decrypts the base64 encoded key and returns the plain text.
        
        :param base64_encoded_key: The encrypted key in base64 encoding.
        :return: The decrypted plain text as a string.
        """
        try:
            encrypted_key = base64.b64decode(base64_encoded_key)
            iv = encrypted_key[:AES.block_size]  # Extract the IV
            cipher_text = encrypted_key[AES.block_size:]  # Extract the ciphertext
            cipher = AES.new(iv, AES.MODE_CBC, iv)  # Recreate the cipher with the extracted IV
            padded_plain_text = cipher.decrypt(cipher_text)
            plain_text = unpad(padded_plain_text, AES.block_size)
            logger.info("Successfully Decrypted the Creds")
            return plain_text.decode('utf-8')
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to Decrypt the Creds Reason: {error} -> TRACEBACK : {traceback.format_exc()}")
            raise
    
    def get_plain_text(self, base64_encoded_key) -> str:
        """
        Public method to get the decrypted plain text.

        :param base64_encoded_key: The encrypted key in base64 encoding.
        :return: The decrypted plain text as a string.
        """
        return self.__decrypt_pass(base64_encoded_key)