"""
Implements various utilities.
"""
import hmac, hashlib, base64


def generate_signature(operation, timestamp, secret_access_key):
    """Return a signature."""
    my_sha_hmac = hmac.new(secret_access_key,
            'AWSMechanicalTurkRequester' + operation + timestamp, hashlib.sha1)
    my_b64_hmac_digest = base64.encodestring(my_sha_hmac.digest()).strip()
    return my_b64_hmac_digest
