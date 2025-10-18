"""
Generate SSL certificates for secure WebSocket communication.
"""
from OpenSSL import crypto
from datetime import datetime, timedelta
import os

def generate_self_signed_cert(
    cert_file="cert.pem",
    key_file="key.pem",
    country="US",
    state="State",
    locality="Locality",
    org="Organization",
    org_unit="Org Unit",
    common_name="localhost"
):
    # Generate key
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    # Generate certificate
    cert = crypto.X509()
    cert.get_subject().C = country
    cert.get_subject().ST = state
    cert.get_subject().L = locality
    cert.get_subject().O = org
    cert.get_subject().OU = org_unit
    cert.get_subject().CN = common_name
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for one year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # Save certificate and private key
    with open(cert_file, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open(key_file, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

    print(f"Generated SSL certificate {cert_file} and private key {key_file}")

if __name__ == '__main__':
    generate_self_signed_cert()