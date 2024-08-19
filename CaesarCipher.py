def encrypt(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            shift_amount = shift % 26
            base = ord('A') if char.isupper() else ord('a')
            new_char = chr((ord(char) - base + shift_amount) % 26 + base)
            result += new_char
        else:
            result += char
    return result

def decrypt(text, shift):
    return encrypt(text, -shift)

def main():
    print("Caesar Cipher Encryption/Decryption")
    mode = input("Enter mode (encrypt/decrypt): ").strip().lower()
    if mode not in ['encrypt', 'decrypt']:
        print("Invalid mode. Please enter 'encrypt' or 'decrypt'.")
        return
   
    message = input("Enter the message: ")
    try:
        shift = int(input("Enter the shift value (integer): "))
    except ValueError:
        print("Invalid shift value. Please enter an integer.")
        return
   
    if mode == 'encrypt':
        encrypted_message = encrypt(message, shift)
        print(f"Encrypted message: {encrypted_message}")
    elif mode == 'decrypt':
        decrypted_message = decrypt(message, shift)
        print(f"Decrypted message: {decrypted_message}")

if __name__ == "__main__":
    main()
